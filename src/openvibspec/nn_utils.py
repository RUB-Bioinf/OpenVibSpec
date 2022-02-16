import os
import socket
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback


# #############################
# Canary Interrupt
# #############################
class CanaryInterruptCallback(Callback):
    """
    This extends the Keras Callback.
    Add this to your model during training to use.

    Upon training starting, a 'canary_interrupt.txt' file is created and this Callback keeps track of it.
    When you delete this file, the training is ended after the current epoch completes.
    You can use this class to manually delete the interrupt file to shut down the training without canceling it and run subsequent functions in your script (eg. saving the weights).

    The canary file is automatically deleted when the training ends.

    Fields to use:
     - active: bool. While true, this callback is active and monitors the state of the canary file.
     - shutdown_source: readonly bool. This param is true, if the canary was triggered and the training stopped.

    .. note::
        Created by Nils Förster.

        Packages Required: os
    """

    def __init__(self, path: str, starts_active: bool = True, label: str = None,
                 out_stream=sys.stdout):
        """
        Constructor for this class.
        Use this function to set up a new canary callback.

        Created by Nils Förster.

        :param path: The path where to save your canary file.
        :param starts_active: If True, the callback will check for file deletion. Default: True.
        :param label: A custom text that will be printed into the canary file. Can be None. Default: None.
        :param out_stream: The outstream used by this Callback. Default: sys.stdout.

        :type path: str
        :type starts_active: bool
        :type label: str
        :type out_stream: TextIOWrapper

        :returns: An instance of this object.
        :rtype: CanaryInterruptCallback

        .. note:: Read the class doc for more info on fields and usage.
        """

        super().__init__()
        self.active: bool = starts_active
        self.label: str = label
        self.shutdown_source: bool = False
        self.out_stream = out_stream

        os.makedirs(path, exist_ok=True)

        self.__canary_file = path + os.sep + 'canary_interrupt.txt'
        if os.path.exists(self.__canary_file):
            os.remove(self.__canary_file)

        f = open(self.__canary_file, 'w')
        f.write(
            'Canary interrupt for CNN training started at ' + gct() + '.\nDelete this file to safely stop your '
                                                                      'training.')
        if self.label is not None:
            f.write('\nLabel: ' + str(self.label).strip())
        f.write('\n\nCreated by Nils Foerster.')
        f.close()

        print('Placed canary file here:' + str(self.__canary_file), file=self.out_stream)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(logs)
        if self.active:
            if not os.path.exists(self.__canary_file):
                print('Canary file not found! Shutting down training!', file=self.out_stream)
                self.shutdown_source = True
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        if os.path.exists(self.__canary_file):
            os.remove(self.__canary_file)


# #############################
# Live Plotting
# #############################
class PlotTrainingLiveCallback(Callback):
    """
    This extends the Keras Callback.
    Add this to your model during training to use.

    This Callback automatically plots and saves specified training metrics on your device and updates the plots at the end of every epoch.
    See the static field 'supported_formats' for a list of all available file formats programmatically.
    Supported image formats are: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.

    You can also export the plots as raw .csv files.
    This callback also supports LaTeX compatible tikz plots.

    This Callback also calculates the training time of each epoch and can plot them as well.
    Based on that, the Callback can calculate the training ETA.


    .. note::
        Created by Nils Förster.
        
        Packages Required: time, os, socket, matplotlib
    """

    supported_formats = ['.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg', '.svgz',
                         '.tif', '.tiff', '.csv', '.tex']
    """
    Lists all file formats supported by this Callback.
    """

    def __init__(self, out_dir: str, out_extensions: [str] = ['.png', '.pdf', '.svg', '.csv', '.tex'],
                 label: str = None, save_timestamps: bool = True, epochs_target: int = None,
                 metrics_names: [str] = None, plot_eta_extra: bool = True, plot_dpi: int = 400,
                 plot_transparency: bool = True):
        """
        Constructor for this class.
        Use this function to set up a new canary callback.

        Created by Nils Förster.

        :param out_dir: The directory to save the plots
        :param out_extensions: A list of strings that feature all the different file extensions to export the metrics to. By default it's: ['.png', '.pdf', '.svg', '.csv', '.tex']
        :param label: An optional label that can be None. If set, this label is printed in the title of every plot. Default: None.
        :param save_timestamps: If true, the timestamps for every epoch is plotted. Default: True.
        :param epochs_target: Optional argument that can be None. If set, this is the target amount of epochs to use when calulating ETA. Currently unused.
        :param metrics_names: Optional argument. A list of strings of metrics to use (eg. ['loss']). When the model also has a validation counterpart of a given metric, this metric is plotted as well.
        :param plot_eta_extra: If true, saves the timestamps plots in an extra directory in png file format. Default: True.
        :param plot_dpi: The (image) DPI to uses for all plots. Default: 400.
        :param plot_transparency: If True, all plots will feature an alpha channel (if supported). Default: True.

        :type out_dir: str
        :type out_extensions: [str] 
        :type label: str, None
        :type save_timestamps: bool 
        :type epochs_target: int, None
        :type metrics_names: [str], None
        :type plot_eta_extra: bool
        :type plot_dpi: int
        :type plot_transparency: bool 

        :returns: An instance of this object.
        :rtype: PlotTrainingLiveCallback

        .. note:: Read the class doc for more info on fields and usage.
        """

        super().__init__()
        self.label = label
        self.out_extensions = out_extensions
        self.metrics_names = metrics_names
        self.plot_dpi = plot_dpi
        self.plot_transparency = plot_transparency
        self.save_timestamps = save_timestamps
        self.epochs_target = epochs_target
        self.plot_eta_extra = plot_eta_extra
        self.out_dir = out_dir

        self.live_plot_dir = out_dir + 'live_plot' + os.sep
        self.timestamp_file_name = self.live_plot_dir + 'training_timestamps.csv'
        os.makedirs(self.live_plot_dir, exist_ok=True)

        if os.path.exists(self.timestamp_file_name):
            os.remove(self.timestamp_file_name)

        self.epoch_start_timestamp = time.time()
        self.epoch_duration_list = []
        self.host_name = str(socket.gethostname())

        self.epochCount = 0
        self.history = {}

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        self._write_timestamp_line('Training start;' + gct())
        self._write_timestamp_line('Epoch;Timestamp')

        if self.metrics_names is None:
            self.metrics_names = self.model.metrics_names

        for metric in self.metrics_names:
            self.history[metric] = []
            self.history['val_' + metric] = []

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        self._write_timestamp_line('Training finished;;' + gct())
        self._plot_training_time()
        self._plot_training_history_live()

    def on_epoch_begin(self, epoch, logs={}):
        super().on_epoch_begin(logs)
        self.epochCount = self.epochCount + 1
        self.epoch_start_timestamp = time.time()
        self._write_timestamp_line()

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(logs)
        t = int(time.time() - self.epoch_start_timestamp)
        self.epoch_duration_list.append(t)

        for metric in self.metrics_names:
            val = 'val_' + metric
            self.history[metric].append(logs[metric])
            self.history[val].append(logs[val])

        self._plot_training_time()
        self._plot_training_history_live()

        if self.plot_eta_extra:
            self._plot_training_time(p_out_dir=self.out_dir, png_only=True)

    def _write_timestamp_line(self, line=None):
        if not self.save_timestamps:
            return

        try:
            f = open(self.timestamp_file_name, 'a')
            if line is None:
                line = str(self.epochCount) + ';' + gct()

            f.write(line + '\n')
            f.close()
        except Exception as e:
            # TODO print stacktrace
            pass

    def _plot_training_time(self, p_out_dir: str = None, png_only: bool = False):
        # Plotting epoch duration
        if p_out_dir is None:
            p_out_dir = self.live_plot_dir

        for extension in self.out_extensions:
            if png_only:
                extension = '.png'

            self._save_metric(data_name='training_time', extension=extension, title='Model Training Time',
                              data1=self.epoch_duration_list, y_label='Duration (Sec.)', p_out_dir=p_out_dir)

    def _save_metric(self, data_name: str, title: str, extension: str, data1: [float], data2: [float] = None,
                     y_label: str = None, p_out_dir: str = None):
        if p_out_dir is None:
            p_out_dir = self.live_plot_dir

        extension = extension.lower().strip()
        if not extension.startswith('.'):
            extension = '.' + extension

        if extension == '.csv':
            self._save_csv_metric(data_name=data_name, data_label=y_label, data1=data1, data2=data2)
            return
        if extension == '.tex':
            self._save_tex_metric(data_name=data_name, title=title, data_label=y_label, data1=data1, data2=data2)
            return

        if self.label is not None:
            title = title + ' [' + self.label + ']'

        plt.title(title)
        plt.ylabel(data_name)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.plot(data1)

        if data2 is not None:
            plt.plot(data2)
            plt.legend(['Train', 'Validation'], loc='best')

        plt.savefig(p_out_dir + data_name + '_live' + extension, dpi=self.plot_dpi,
                    transparent=self.plot_transparency)
        plt.clf()

    def _save_csv_metric(self, data_name: str, data_label: str, data1, data2=None):
        f_name = self.live_plot_dir + data_name + '_live.csv'
        f = open(f_name, 'w')

        f.write('Epoch;' + data_label)
        if data2 is not None:
            f.write(';Validation ' + data_label)
        f.write(';\n')

        for i in range(len(data1)):
            f.write(str(i + 1) + ';' + str(data1[i]))
            if data2 is not None:
                f.write(';' + str(data2[i]))
            f.write(';\n')

    def _save_tex_metric(self, data_name: str, title: str, data_label: str, data1, data2=None) -> str:
        data = [data1]
        titles = ['Training']
        colors = ['blue']

        min_y = min(data1)
        max_y = max(data1)

        if data2 is not None:
            data.append(data2)
            titles.append('Validation')
            colors.append('orange')

            min_y = min(min_y, min(data2))
            max_y = max(max_y, max(data2))

        min_y = max(min_y - 0.1337, 0)
        max_y = min(max_y + 0.1337, 1)

        out_text = get_plt_as_tex(data_list_y=data, plot_titles=titles, plot_colors=colors, title=title,
                                  label_y=data_label, max_x=len(data1), min_x=1, max_y=max_y, min_y=min_y)

        f_name = self.live_plot_dir + data_name + '_live.tex'
        f = open(f_name, 'w')
        f.write(out_text)
        f.close()

        return out_text

    def _plot_training_history_live(self):
        # Plotting epoch duration
        for metric in self.metrics_names:
            val = 'val_' + metric
            m = metric.capitalize()
            title = 'Model: ' + m

            for extension in self.out_extensions:
                data1 = self.history[metric]
                data2 = None
                if val in self.history:
                    data2 = self.history[val]

                self._save_metric(data_name=metric, extension=extension, title=title, y_label=m, data1=data1,
                                  data2=data2)


# ###############################
# STATIC FIELDS
# ###############################

pgf_plot_colors = [
    'red',
    'green',
    'blue',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'gray',
    'darkgray',
    'lightgray',
    'brown',
    'lime',
    'olive',
    'orange',
    'pink',
    'purple',
    'teal',
    'violet',
    'white']


# ###############################
# OTHER UTIL FUNCTIONS
# ###############################

def gct(raw: bool = False) -> [str, datetime]:
    """
    Gets the current time as a formated string or datetime object.
    Shortcut function.

    Created by Nils Förster.

    :param raw: An optional parameter. If True, the time will be returned as a datetime object. Default: False.

    :type raw: bool

    :returns: The current time. Either as a formated string or datetime object.
    :rtype: datetime,str

    .. note:: Required packages: datetime
    """
    n = datetime.now()
    if raw:
        return n
    return n.strftime("%d/%m/%Y %H:%M:%S")


def get_time_diff(start_time: datetime) -> str:
    """
    Calculates the time difference from a given datetime object and the current time this function is being called.

    Created by Nils Förster.

    :param start_time: The timestamp to calculate the difference from

    :type start_time: datetime

    :returns: The time difference as a formated string.
    :rtype: str

    .. note:: Required packages: datetime
    """

    diff = datetime.now() - start_time
    minutes = divmod(diff.total_seconds(), 60)

    m: str = str(int(minutes[0]))
    s: str = str(int(minutes[1]))
    if minutes[1] < 10:
        s = '0' + s
    return m + ':' + s


# ###############################
# API TO LaTeX TIKZ
# ###############################

def create_tikz_axis(title: str, label_y: str, label_x: str = 'Epoch', max_x: float = 1.0, min_x: float = 0.0,
                     max_y: float = 1.0, min_y: float = 0.0, tick_count: int = 10,
                     legend_pos: str = 'north west') -> str:
    """
    Sets up a basic tikz plot environment to be used in a LaTeX document.
    This is a helper function; For the true function try #get_plt_as_tex.
    That function fills the tikz plot with graphs.

    Created by Nils Förster.

    :param title: The title to be used in the plot.
    :param label_y: The label for the y axis.
    :param label_x: Optional argument. The label for the x axis. Default: 'Epoch'.
    :param max_x: Optional argument. The maximum span for the x axis. Default: 1.0
    :param min_x: Optional argument. The minimum span for the x axis. Default: 0.0
    :param max_y: Optional argument. The maximum span for the y axis. Default: 1.0
    :param min_y: Optional argument. The maximum span for the y axis. Default: 0.0
    :param tick_count: Optional argument. In how many 'ticks' should the plot be partitioned? Default: 10.
    :param legend_pos: Optional argument. The position of the legend. Default: 'north-west'.

    :type title: str
    :type label_y: str
    :type label_x: str
    :type max_x: float
    :type min_x: float
    :type max_y: float
    :type min_y: float
    :type tick_count: int
    :type legend_pos: str

    :returns: A template for a tikz plot as a string.
    :rtype: str
    """

    max_x = float(max_x)
    max_y = float(max_y)
    tick_count = float(tick_count)

    tick_x = max_x / tick_count
    tick_y = max_y / tick_count
    if min_x + max_x > 10:
        tick_x = int(tick_x)
    if min_y + max_y > 10:
        tick_y = int(tick_y)

    axis_text: str = '\\begin{center}\n\t\\begin{tikzpicture}\n\t\\begin{axis}[title={' + title + '},xlabel={' + label_x + '},ylabel={' + label_y + '},xtick distance=' + str(
        tick_x) + ',ytick distance=' + str(tick_y) + ',xmin=' + str(min_x) + ',xmax=' + str(
        max_x) + ',ymin=' + str(min_y) + ',ymax=' + str(
        max_y) + ',major grid style={line width=.2pt,draw=gray!50},grid=both,height=8cm,width=8cm'
    if legend_pos is not None:
        axis_text = axis_text + ', legend pos=' + legend_pos
    axis_text = axis_text + ']'
    return axis_text


def get_plt_as_tex(data_list_y: [[float]], plot_colors: [str], title: str, label_y: str, data_list_x: [[float]] = None,
                   plot_titles: [str] = None, label_x: str = 'Epoch', max_x: float = 1.0, min_x: float = 0.0,
                   max_y: float = 1.0, min_y: float = 0.0, max_entries: int = 4000, tick_count: int = 10,
                   legend_pos: str = 'north west') -> str:
    """
    Formats a list of given plots in a single tikz axis to be compiled in LaTeX.
    
    This function respects the limits of the tikz compiler.
    That compiler can only use a limited amount of virtual memory that is (to my knowledge not changeable).
    Hence this function can limit can limit the line numbers for the LaTeX document.
    Read this function's parameters for more info.

    This function is designed to be used in tandem with the python library matplotlib.
    You can plot multiple graphs in a single axis by providing them in a list.
    Make sure that the lengths of the lists (see parameters below) for the y and x coordinates, colors and legend labels match.

    Created by Nils Förster.

    :param data_list_y: A list of plots to be put in the axis. Each entry in this list should be a list of floats with the y position of every node in the graph.
    :param plot_colors: A list of strings descibing colors for every plot. Make sure len(data_list_y) matches len(plot_colors).
    :param title: The title to be used in the plot.
    :param label_y: The label for the y axis.
    :param plot_titles: Optional argument. A list of strings containing the legend entries for every graph. If None, no entries are written in the legend. Make sure len(data_list_y) matches len(plot_titles).
    :param data_list_x: Optional argument. A list of plots to be put in the axis. Each entry in this list should be a list of floats with the x position of every node in the graph. If this argument is None, the entries in the argument data_list_y are plotted as nodes in sequential order.
    :param label_x: Optional argument. The label for the x axis. Default: 'Epoch'.
    :param max_x: Optional argument. The maximum span for the x axis. Default: 1.0
    :param min_x: Optional argument. The minimum span for the x axis. Default: 0.0
    :param max_y: Optional argument. The maximum span for the y axis. Default: 1.0
    :param min_y: Optional argument. The maximum span for the y axis. Default: 0.0
    :param tick_count: Optional argument. In how many 'ticks' should the plot be partitioned? Default: 10.
    :param legend_pos: Optional argument. The position of the legend. Default: 'north-west'.
    :param max_entries: Limits the amount of nodes for the plot to this number. This does not cut of the data, but increases scanning offsets. Use a smaller number for faster compile times in LaTeX. Default: 4000.

    :type data_list_y: [[float]]
    :type plot_colors: [str]
    :type title: str
    :type label_y: str
    :type plot_titles: [str]
    :type label_x: str
    :type max_x: float
    :type min_x: float
    :type max_y: float
    :type min_y: float
    :type tick_count: int
    :type legend_pos: str
    :type max_entries: int

    :returns: A fully formated string containing a tikz plot with multiple graphs and and legends. Save this to your device and compile in LaTeX to render your plot.
    :rtype: str

    Examples
    ----------
    Use this example to plot a graph in matplotlib (as plt) as well as tikz:

    >>> history = model.fit(...)
    >>> loss = history.history['loss']
    >>> plt.plot(history_all.history[hist_key]) # plotting the loss using matplotlib
    >>> f = open('example.tex')
    >>> tex = get_plt_as_tex(data_list_y=[loss], title='Example Plot', label_y='Loss', label_x='Epoch', plot_colors=['blue']) # plotting the same data as a tikz axis
    >>> f.write(tex)
    >>> f.close()

    When you want to plot multiple graphs into a single axis, expand the example above like this:

    >>> get_plt_as_tex(data_list_y=[loss, val_loss], title='Example Plot', label_y='Loss', label_x='Epoch', plot_colors=['blue'], plot_titles=['Loss','Validation Loss'])

    When trying to render the tikz plot, make sure to import these LaTeX packages:

    >>> \\usepackage{tikz,amsmath, amssymb,bm,color,pgfplots}

    .. note:: Some manual adjustments may be required to the tikz axis. Try using a wysisyg tikz / LaTeX editor for that. For export use, read the whole tikz user manual ;)
    """

    out_text = create_tikz_axis(title=title, label_y=label_y, label_x=label_x, max_x=max_x, min_x=min_x, max_y=max_y,
                                min_y=min_y, tick_count=tick_count, legend_pos=legend_pos) + '\n'
    line_count = len(data_list_y[0])
    data_x = None
    steps = int(max(len(data_list_y) / max_entries, 1))

    for j in range(0, len(data_list_y), steps):
        data_y = data_list_y[j]
        if data_list_x is not None:
            data_x = data_list_x[j]

        color = plot_colors[j]

        out_text = out_text + '\t\t\\addplot[color=' + color + '] coordinates {' + '\n'
        for i in range(line_count):
            y = data_y[i]

            x = i + 1
            if data_x is not None:
                x = data_x[i]

            out_text = out_text + '\t\t\t(' + str(x) + ',' + str(y) + ')\n'
        out_text = out_text + '\t\t};\n'

        if plot_titles is not None:
            plot_title = plot_titles[j]
            out_text = out_text + '\t\t\\addlegendentry{' + plot_title + '}\n'

    out_text = out_text + '\t\\end{axis}\n\t\\end{tikzpicture}\n\\end{center}'
    return out_text


####


def normalize_np(n: np.ndarray, lower: float = 0.0, upper: float = 255.0) -> np.ndarray:
    """
    Using linear normalization, every entry in a given numpy array between a lower and upper bound.
    The shape of the array can be arbitrary.

    If the lower and upper bound are equal, the reulting array will contain only zeros.

    Created by Nils Förster.

    :param n: An arbitrary numpy array
    :param lower: The lower bound for normalization
    :param upper: The upper bound for normalization

    :type n: np.ndarray
    :type lower: float
    :type upper: float

    :returns: Returns a copy of the array normalized between 0 and 1, relative to the lower and upper bound
    :rtype: np.ndarray

    Examples
    ----------
    Use this example to generate ten random integers between 0 and 100. Then normalize them using this function.

    >>> n=np.random.randint(0,100,10)
    >>> normalize_np(n,0,100)

    """
    nnv = np.vectorize(_normalize_np_worker)
    return nnv(n, lower, upper)


####

def _normalize_np_worker(x: float, lower: float, upper: float):
    if lower == upper:
        return 0

    lower = float(lower)
    upper = float(upper)
    return (x - lower) / (upper - lower)


##########
#### MAIN
##########


if __name__ == "__main__":
    print('There are some util functions for everyone to use within this file, created by Nils Förster. Enjoy. :)')
