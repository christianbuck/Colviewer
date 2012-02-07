#!/usr/bin/env python

import sys
import os
import gtk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from collections import Counter, defaultdict
from pylab import cm
from itertools import izip

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

    cmap = cm.get_cmap('RdYlGn')

class ColViewer:

    def __init__(self, coldir=None):
        self.equalize = False
        self.equalize_cutoff = 20
        self.data_binary = True
        self.n_values = 2
        self.n_bins = 10
        self.n_bins_desired = self.n_bins
        self.tgt2color = {}

        builder = gtk.Builder()
        xmlfile = os.path.join(os.path.split(sys.argv[0])[0], "colviewer_main_window.xml")
        builder.add_from_file(xmlfile)

        self.window = builder.get_object("window")
        print builder.connect_signals(self)

        self.logbuffer = builder.get_object("textbuffer_info") # GtkTextBuffer
        self.append_to_log("initialized")

        self.coldir = coldir
        if not self.coldir:
            self.coldir = os.getcwd()

        self.filelist = builder.get_object("list_files") # GtkListStore
        self.fill_filelist()

        # scale and spinbutton to set number of bins
        self.n_bins_scale = builder.get_object("hscale_nbins") # GtkHScale
        self.n_bins_spinb = builder.get_object("spinbutton_nbins") # GtkSpinButton
        self.set_nbins_sensible()

        # label showing target filename
        self.target_label = builder.get_object("label_target_filename")

        self.barchart_box = builder.get_object("box_barchart") # GtkVBox
        #zeroCounts   = (0, 0)
        #oneCounts    = (0, 0)
        #self.data = [zeroCounts, oneCounts]
        self.init_barchart()

    ### GUI Tools ###
    def set_target_filename(self, filename):
        self.target_label.set_text(os.path.split(filename)[-1])
        self.target_label.set_tooltip_text(filename)

    def update_barchart(self):
        self.histogram()
        self.plot_barchart()
        self.plot_errorbars()
        self.bar_canvas.draw()

    def fill_filelist(self):
        files = os.listdir(self.coldir)
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            self.filelist.append([filename])

    def append_to_log(self, text):
        self.logbuffer.insert(self.logbuffer.get_end_iter(),'# ' + text + "\n")

    def set_nbins_sensible(self):
        """ activate or deactivate interface elements that only make sense
            for data with binary / non-binary values """
        self.n_bins_scale.set_sensitive(not self.data_binary)
        self.n_bins_spinb.set_sensitive(not self.data_binary)

    def show(self):
        self.window.show()

    ### Plotting ###
    def init_barchart(self):
        fig = Figure(facecolor='w')
        ax = fig.add_subplot(211,axisbg=NiceColors.lightgrey)
        self.bar_ax = ax

        self.err_ax = fig.add_subplot(212,axisbg=NiceColors.lightgrey)
        #self.plot_barchart(data, ax)

        self.bar_canvas = FigureCanvas(fig)  # a gtk.DrawingArea
        self.bar_canvas.show()
        self.barchart_box.pack_start(self.bar_canvas)

    def get_layers(self):
        " return one layer for each target in sorted order "
        for tgt in sorted(self.data.keys()):
            cnt = Counter(bin_id for val, bin_id in self.data[tgt])
            layer = [cnt[b] for b in range(self.n_bins)]
            if self.equalize:
                mult = self.get_equalize_multiplicator()
                assert len(mult) == len(layer)
                layer = [c*f for c, f in zip(layer, mult)]
            yield (tgt, layer)

    def get_locations(self, limits):
        " locations of bars are in the middle between lower and upper bound"
        return [(u+l)/2.0 for l,u in izip(limits[:-1], limits[1:])]

    def get_width(self, locations=None, gap_factor=1.0):
        if locations == None:
            locations = self.get_locations(self.limits)
        if len(locations) < 3:
            return 1.0
        return gap_factor * min( (l2-l1) for l1, l2 in
            zip(locations[:-1], locations[1:]) )

    def get_equalize_multiplicator(self):
        return [1./float(c) if c > self.equalize_cutoff
                else 0. for c in self.col_counts]

    def plot_errorbars(self):
        #print len(locations) , len(self.cols)
        self.err_ax.cla()
        assert len(self.locations) == len(self.cols)
        err = [np.std(c) for c in self.cols]
        means = [np.mean(c) for c in self.cols]
        self.err_ax.errorbar(self.locations, means, yerr=err, ecolor="black")
        self.err_ax.set_xbound( self.bar_ax.get_xbound() )
        #for l,m,e in izip(self.locations, means, err):
        #    self.err_ax.errorbar(l, m, yerr=e, ecolor="black")

    def plot_barchart(self):
        self.bar_ax.cla()
        assert len(self.limits) == self.n_bins + 1, "mismatch n_bins and limits"
        self.locations = self.get_locations(self.limits)
        width = self.get_width(self.locations, 0.8)
        colormap = NiceColors.cmap
        bottom = [0] * self.n_bins
        for tgt, layer in self.get_layers():
            assert len(layer) == len(bottom)
            color = colormap(self.normalize(tgt, self.min_tgt, self.max_tgt))
            self.bar_ax.bar(self.locations, layer, width,
                            color=color, bottom=bottom, linewidth=0,
                            align='center')
            bottom = [b+c for b,c in zip(bottom, layer)]

        #ax.set_xticks((0.0,1.0))
        #ax.set_yticks([0,25,50,75,100],minor=False)
        #ax.yaxis.grid(True)
        #ax.set_xticklabels(('0', '1'))
        if self.equalize:
            self.bar_ax.set_ybound(lower=0.0, upper=1.2)

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
        self.n_bins_desired = int(adjustment.get_value())
        self.update_barchart()

    def on_check_equal_toggled(self, checkbutton):
        self.equalize = checkbutton.get_active()
        self.update_barchart()

    ### Data Handling ###
    def load_bincol(self, filename):
        self.data = map(int, map(float,open(filename)))

    def load_col(self, filename):
        fname = os.path.join(self.coldir, filename)
        self.raw_data = map(float, open(fname))
        if len(self.raw_data) > len(self.target):
            sys.stdout.write("WARN: more data than targets (%s vs %s)\n"
                             %(len(data), len(targets)))
            self.raw_data = self.raw_data[:len(self.target)]
        self.n_values = len(set(self.raw_data))
        self.data_binary = self.n_values == 2

        self.set_nbins_sensible()
        self.append_to_log("loaded %s lines containing %s uniq values" %(len(self.raw_data), self.n_values))
        if self.data_binary:
            self.append_to_log("%s zeros, %s ones" %(self.raw_data.count(0.0), self.raw_data.count(1.0)))
        self.update_barchart()

    def read_target(self, filename):
        # self.target = map(int, map(float, open(filename)))
        self.target = map(float, open(filename))
        self.n_classes = len(set(self.target))
        self.append_to_log("loaded target: %s lines containing %s classes"
                           %(len(self.target), self.n_classes))
        self.set_target_filename(filename)
        self.min_tgt = min(self.target)
        self.max_tgt = max(self.target)

        #self.reset_data()
        # redo histogram if data already read

    def reset_data(self):
        self.data = dict( (tgt, []) for tgt in set(self.target) )

    ### Histogram ###
    def find_bin(self, val, bins):
        """ bins are given as bins=[x1 x2 x3 x4]
            returns largest i such that val <= bins[i+1]
        """
        nbins = len(bins)
        assert nbins >= 2, "%s does not specify proper intervals" %(bins)
        assert val <= bins[-1], "value larger than end of last interval"
        assert val >= bins[0], "value smaller than start of first interval"
        for i in range(nbins-1):
            if val < bins[i+1]:
                assert val >= bins[i]
                return i
        return nbins - 2

    def histogram(self):
        """ make histogram and update self.data with bin infos """
        self.reset_data()
        self.n_bins = min(self.n_values, self.n_bins_desired)
        n, bins, patches = self.bar_ax.hist(self.raw_data, self.n_bins)
        self.col_counts = n
        self.cols = [[] for c in range(self.n_bins)]
        for val, tgt in izip(self.raw_data, self.target):
            bin_id = self.find_bin(val, bins)
            assert bin_id <= self.n_bins, "%s > %s" %(bin_id, self.n_bins)
            self.data[tgt].append((val, bin_id))
            self.cols[bin_id].append(tgt)
        self.limits = bins

    def make_columns(self, nbins):
        cols = []
        for tgt in sorted(set(self.target)):
            color = self.tgt2color[tgt]
            bars = self.data[tgt]
            cols.append(color, bars)

        #for idx, val in enumerate(self.raw_data):
        #    binidx = 1
        #    while bins[binidx] < val:
        #        binidx += 1
        #        assert binidx < len(bins)
        #    self.data[self.target[idx]][binidx-1] += 1
        ## scale to percent
        #factor = 100./len(self.raw_data)
        #self.data = [[factor*val for val in d] for d in self.data]

    def equalize_data(self, data, limit=100.):
        heights = map(sum,zip(*data))
        factors = [limit/height for height in heights]
        return [[factor*val for factor,val in zip(factors,d)] for d in data]

    def normalize(self, value, min_val=0.0, max_val=1.0):
        return float(value-min_val)/(max_val-min_val)

    def color_column(values, colormap, min_val=0.0, max_val=1.0):
        """ values are targets for one column """
        bottom = [0]
        height = [0]
        colors = [0]
        counts = Counter(values)
        for val, count in sorted(counts.items()):
            bottom.append(bottom[-1] + count)
            height.append(count)
            colors.append(colormap(self.normalize(value, min_val, max_val)))
        return zip(height, bottom, colors)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '-target', action='store', dest='target', required=True)
    parser.add_argument('-n', action='store', help='n bins', type=int, dest="nbins", default=5)
    #parser.add_argument('-l', action='store_true', help='logscale y-axis', dest="log", default=False)
    parser.add_argument('-coldir', action='store', help='column dir')
    args = parser.parse_args(sys.argv[1:])

    colviewer = ColViewer(coldir=args.coldir)
    colviewer.read_target(args.target)
    colviewer.show()
    gtk.main()
