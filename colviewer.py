#!/usr/bin/env python

import sys
import os
import gtk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas

from numpy import arange, sin, pi

class NiceColors:
    lightgrey = '0.85'
    darkgrey = '0.6'
    nicegreen = '#A7BD5B'
    nicered = '#DC574E'
    niceyellow = '#F7E4A2'
    niceblue = '#8DC7B8'
    niceorange = '#ED9355'
    
    twoclass = [nicegreen, nicered]
    blushed = ['#7D8258','#CFB97C','#E5894F','#C75C44','#A63E38']
    autumn  = ['#B8B71F','#737A14','#AF6413','#A01C09','#50110A']

class ColViewer:

    def __init__(self):
        self.equalize = False
        self.data_binary = True
        self.n_values = 2
        self.n_bins = 2
        
        builder = gtk.Builder()
        xmlfile = os.path.join(os.path.split(sys.argv[0])[0], "colviewer_main_window.xml")
        builder.add_from_file(xmlfile) 
  
        self.window = builder.get_object("window")
        print builder.connect_signals(self)
        
        self.logbuffer = builder.get_object("textbuffer_info") # GtkTextBuffer
        self.append_to_log("initialized")
        
        self.filelist = builder.get_object("list_files") # GtkListStore
        self.fill_filelist()

        # scale and spinbutton to set number of bins
        self.n_bins_scale = builder.get_object("hscale_nbins") # GtkHScale
        self.n_bins_spinb = builder.get_object("spinbutton_nbins") # GtkSpinButton
        self.set_nbins_sensible()
        
        # label showing target filename
        self.target_label = builder.get_object("label_target_filename")
        
        self.barchart_box = builder.get_object("box_barchart") # GtkVBox
        zeroCounts   = (0, 0)
        oneCounts    = (0, 0)
        self.data = [zeroCounts, oneCounts]
        self.barchart(self.data)

    ### GUI Tools ###
    def set_target_filename(self, filename):
        self.target_label.set_text(os.path.split(filename)[-1])
        self.target_label.set_tooltip_text(filename)

    def update_barchart(self):
        self.plot_barchart(self.data, self.bar_ax)
        self.bar_canvas.draw()

    def fill_filelist(self):
        files = os.listdir(os.getcwd())
        for filename in files:
            self.filelist.append([filename])
            
    def append_to_log(self, text):
        self.logbuffer.insert(self.logbuffer.get_end_iter(),'# ' + text + "\n")
        
    def set_nbins_sensible(self):
        self.n_bins_scale.set_sensitive(not self.data_binary)
        self.n_bins_spinb.set_sensitive(not self.data_binary)
    
    def show(self):
        self.window.show()
    
    ### Plotting ###
    def barchart(self, data):
        fig = Figure(facecolor='w')
        ax = fig.add_subplot(111,axisbg=NiceColors.lightgrey)
        self.bar_ax = ax
        self.plot_barchart(data, ax)
        
        self.bar_canvas = FigureCanvas(fig)  # a gtk.DrawingArea
        self.bar_canvas.show()
        self.barchart_box.pack_start(self.bar_canvas)
    
    def plot_barchart(self, data, ax, width=0.5):
        ax.cla()
        if self.equalize:
            data = self.equalize_data(data)
            
        locations = (0-width/2, 1-width/2)    # the x locations for the groups

        colormap = NiceColors.twoclass
        if len(data) > 2:
            colormap = NiceColors.autumn
            assert len(colormap) >= len(data), 'more classes that colors'
            
        for clsidx, counts in enumerate(data):
            color = colormap[clsidx]
            if clsidx == 0:
                ax.bar(locations, counts, width, color=color)
            else:
                bottoms = map(sum,zip(*data[:clsidx]))
                ax.bar(locations, counts, width, color=color, bottom=bottoms)
        
        ax.set_xticks((0.0,1.0))
        ax.set_yticks([0,25,50,75,100],minor=False)
        ax.yaxis.grid(True)
        ax.set_xticklabels(('0', '1'))
        ax.set_ybound(lower=0, upper=100)
    
    ### Signal Handling ###
    def on_window_destroy(self, widget, data=None):
        gtk.main_quit()
        
    def on_treeview_filenames_cursor_changed(self, treeview):
        treeselection = treeview.get_selection()
        (treemodel, treeiter) = treeselection.get_selected()
        col = 0
        filename = treemodel.get_value(treeiter, col)
        self.load_col(filename)
        
    def on_adjustment_nbins_value_changed(self, adjustment):
        self.n_bins = int(adjustment.get_value())
    
    def on_check_equal_toggled(self, checkbutton):
        self.equalize = checkbutton.get_active()
        self.update_barchart()

    ### Data Handling ###
    def load_bincol(self, filename):
        self.data = map(int, map(float,open(filename)))
    
    def load_col(self, filename):
        self.raw_data = map(float, open(filename))
        if len(self.raw_data) > len(self.target):
            self.raw_data = self.raw_data[:len(self.target)]
        self.n_values = len(set(self.raw_data))
        if  self.n_values == 2:
            self.data_binary = True
            
        self.set_nbins_sensible()
        self.append_to_log("loaded %s lines containing %s uniq values" %(len(self.raw_data), self.n_values))
        if self.data_binary:
            self.append_to_log("%s zeros, %s ones" %(self.raw_data.count(0.0), self.raw_data.count(1.0)))
        self.histogram()
        self.update_barchart()

    def read_target(self, filename):
        self.target = map(int, map(float, open(filename)))
        self.n_classes = len(set(self.target))
        self.append_to_log("loaded %s lines containing %s classes" %(len(self.target), self.n_classes))
        self.set_target_filename(filename)
    
    ### Histogram ###
    def histogram(self):
        nbins = min(self.n_values, self.n_bins)
        n, bins, patches = self.bar_ax.hist(self.raw_data, nbins)
        self.data = [[0]*nbins for c in range(self.n_classes)]
        for idx, val in enumerate(self.raw_data):
            binidx = 1
            while bins[binidx] < val:
                binidx += 1
                assert binidx < len(bins)
            self.data[self.target[idx]][binidx-1] += 1
        # scale to percent
        factor = 100./len(self.raw_data)
        self.data = [[factor*val for val in d] for d in self.data]

    def equalize_data(self, data, limit=100.):
        heights = map(sum,zip(*data))
        factors = [limit/height for height in heights]
        return [[factor*val for factor,val in zip(factors,d)] for d in data]
    
if __name__ == "__main__":
    colviewer = ColViewer()
    
    colviewer.read_target(sys.argv[1])
    
    colviewer.show()
    gtk.main()