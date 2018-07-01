import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.definitions import ROOT_DIR

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA



def __list_files_by_name(path, similarity, variable_position,
                         knn=False, shrink=False, alpha=False, beta=False):
    """
    knn, shrink, alpha: filters files with that knn if give.
    example:

    files = list_files(path, name, knn="50", alpha="0.2" ):
    """

    files = list()
    changing_vars = list()

    filenames = os.listdir(path)
    assert len(filenames)>0 , " No files found at path"

    for filename in sorted(filenames):

        if similarity in filename and filename.endswith(".csv"):

            items = filename.replace(" ", "").replace(".csv", "").split('-')
            if similarity:
                if items[2] != similarity:
                    continue
            if knn:
                if items[3] != knn:
                    continue
            if shrink:
                if items[4] != shrink:
                    continue
            if alpha:
                if str(items[5]) != str(alpha):
                    continue
            if beta:
                if str(items[6]) != str(beta):
                    continue
            files.append(filename)

    for file in sorted(files):
        items = file.replace(" ", "").replace(".csv", "").split('-')
        changing_vars.append(items[variable_position+3])

    return files, changing_vars

def __display_one(array_to_display, algorithm_name, metric, grouped_by_what, xticks, category_num,
                  show=True, save=False):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    x = np.arange(len(array_to_display))
    y = np.array(array_to_display)

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)

    ax.set_ylim(0, max(array_to_display) * 1.1)

    plt.plot(x, y)
    for i, j in zip(x, y):
        ax.annotate('%.3f' % j, xy=(i, j))

    if not category_num:
        category_num = " all "
    plt.title(algorithm_name + "\ncat=" + str(category_num), fontdict=font)

    plt.xticks(np.arange(len(xticks)), [x.replace(",","\n") for x in xticks])

    plt.ylabel(metric, fontdict=font)
    plt.xlabel(grouped_by_what, fontdict=font)
    plt.grid()
    if save:
        plt.savefig(ROOT_DIR+"/results/img/"+algorithm_name+"-"+grouped_by_what+"-"+metric+"-cat="+category_num, dpi=150,)
    if show:
        plt.show()
    plt.close(fig)



def __plot_one_by_one(results, algorithm_name, num_cat, xticks_vars, variable_name,save,show ):
    metrics = ['clicks_artists', 'clicks_tracks', 'ndcg_artists', 'ndcg_tracks', 'precision_artists', 'precision_tracks']
    for metric in metrics:

        __display_one(array_to_display=results[metric],
                        metric=metric,
                        algorithm_name=algorithm_name,
                        category_num=num_cat,
                        xticks=xticks_vars,
                        grouped_by_what=variable_name,
                        show=show,
                        save=save)




res = ['clicks_artists','clicks_tracks','ndcg_artists','ndcg_tracks','precision_artists','precision_tracks']
target = res[2]


def plot_results_by_name( algorithm_name,variable_position, variable_name ,  path_csvs=ROOT_DIR+'/results/tests', num_cat=False,
                  knn=False, shrink=False, alpha=False, beta=False,
                  show=True, save=False,
                  verbose=False):
    """

    :param path_csvs:       path to results_csvs
    :param algorithm_name:  name of the algorithm (the one of the csvs)
    :param variable_position:  Which one is the variable you are plotting (0,1,2,3)
    :param num_cat:         int num 1/9 or leave it False for all categories
    :param knn:             variable
    :param shrink:          variable
    :param alpha:           variable
    :param beta:            variable  /// give the constant value for the variables and leave the one to plot at False

    :param show:            if to show the plots
    :param save:            if to save the png(s)
    :param verbose:         by default at False

    example:
                import utils.plot_results_csv as plot
                plot.plot_results_by_name(  path_csvs='../results/test2/',
                                            algorithm_name="tversky", variable_name='alpha',
                                            knn='100',shrink='0', beta='1',
                                            variable_position=2,
                                            show=True, save=True,
                                            verbose=True)

    """


    if not num_cat:
        num_cat = "ALL"

    if verbose: print("[searching in "+os.path.dirname(os.path.abspath(__file__))+path_csvs+" ]")
    files, xticks_vars = __list_files_by_name(path_csvs, variable_position=variable_position,
                                              similarity=algorithm_name,
                                              knn=knn, shrink=shrink, alpha=alpha,beta=beta)

    # for f in files:
    #     print(f)

    assert len(files)>0 , " wrong parameters, no files have what you said"

    results = {'clicks_artists':[0]*len(files), 'clicks_tracks':[0]*len(files),
               'ndcg_artists':[0]*len(files),   'ndcg_tracks':[0]*len(files),
               'precision_artists':[0]*len(files),  'precision_tracks':[0]*len(files)   }


    for res in results:
        i = 0
        for file in files:
            df = pd.read_csv(path_csvs+file, sep='\t')
            j = iter(range(1, len(df.columns) + 1))
            df.columns = [x if not x.startswith('Unnamed') else str(next(j)) for x in df.columns]
            if num_cat == "ALL":
                df = df[(df['1'] == 'mean')]
            else:
                df = df[(df['1'] == 'cat'+str(num_cat))]
            results[res][i] = df.iloc[0][res]
            i += 1

    __plot_one_by_one(results=results,algorithm_name=algorithm_name, num_cat=num_cat,
                      xticks_vars=xticks_vars, variable_name=variable_name, save=save, show=show)





def plot_graph_from_simones_file(path_to_file_and_filename, pdf_name, show=False, all_in_one=True):

    if not os.path.exists(path_to_file_and_filename):
        raise FileExistsError("il file non esiste. controlla la path: "+path_to_file_and_filename)

    lines = tuple(open(path_to_file_and_filename, 'r'))

    metrics = ['rp', 'ndcg', 'click']
    df = pd.DataFrame()
    first = False
    cat = None
    titolo = ""
    titolo2 = ""

    if lines[0].startswith("###"):
        titolo = ""
        start = 0
    else:
        titolo = lines[0]
        start = 1



    for line in lines[start:]:

        # read the variable line, if last line was the first, now i have the variable lines
        if first:
            variables_str = line.replace("\n","")
            variables = line.replace(" ", "").replace("\n","").split(sep=',')
            var_dict = dict( [x.split("=") for x in variables])
            first = False

        # read the start of the block
        elif line.startswith("###"):
            first = True

        elif line.startswith("---"):
            titolo2 = line.replace("-","").replace(" ","").replace("\n","")

        # read the category
        elif line.startswith("##") and line[2].isdigit() and line[3].isdigit():
            cat = line[2]+line[3]

        elif line.startswith("##") and line[2].isdigit() and not line[3].isdigit():
            cat = line[2]

        # read the results
        else:
            rp, ndcg, click = line.replace(" ", "").replace("\n","").split(sep=',')

            result_dict =  dict({'cat': cat, 'rp': rp, 'ndcg': ndcg, 'click': click,
                                 'vars_str':variables_str}, **var_dict )

            df = df.append(result_dict, ignore_index=True)

    pdf_file = PdfPages(pdf_name+'.pdf')


    if all_in_one:
        for prog in range(1,11):
            vals = df[df.cat == str(prog)]

            rp_array = [float(x) for x in vals['rp'].values.ravel()]

            click_array = [float(x) for x in vals['click'].values.ravel()]

            ndcg_array = [float(x) for x in vals['ndcg'].values.ravel()]

            if not (len(rp_array)==0 or len(click_array)==0 and len(ndcg_array)==0):

                xticks_todisp = [float(x.split('=')[1].replace(" ", "")) for x in vals['vars_str'].values]

                # x_ticks_filtered = list(xticks_todisp.copy())
                # if len(xticks_todisp)>3:
                #     jump = int(len(xticks_todisp)/3)
                #     print(jump)
                #     for i in range(len(xticks_todisp)):
                #         if i%jump != 0 and i!=0 and i!=len(xticks_todisp)-1:
                #             x_ticks_filtered[i]='.'
                #         else:
                #             if xticks_todisp[i] is float:
                #                 x_ticks_filtered[i] = str("%.2f" % xticks_todisp[i])
                #             elif xticks_todisp[i] is int:
                #                 x_ticks_filtered[i] = xticks_todisp[i]
                # print(xticks_todisp)
                # print(x_ticks_filtered)

                y_label = vals['vars_str'].values[0].split('=')[0].replace(" ", "")

                __display_from_three_arrays(rp_array=rp_array, click_array=click_array, ndcg_array=ndcg_array,
                                        titolo=titolo2+"\ncat"+str(prog),
                                        category_num=prog, x_ticks = xticks_todisp, x_ticks_filtered=None,
                                        y_label=y_label,
                                        show = show, save = False , pdf_file =pdf_file)

    else:

        ##### print the graph for each cat, for each metric
        for prog in range(1,11):
            vals = df[df.cat == str(prog)]

            for metric in metrics:
                if len(vals[metric].values)>0:

                    to_disp = [float(x) for x in vals[metric].values.ravel() ]

                    xticks_todisp = [ float(x.split('=')[1].replace(" ","")) for x in  vals['vars_str'].values ]

                    __display_from_array(df = df,
                                      array_to_display=to_disp, titolo=titolo2+titolo+" cat"+str(prog),
                                      metric=metric, category_num=prog, x_ticks = xticks_todisp,
                                      show = show, save = False , pdf_file =pdf_file)


    pdf_file.close()


def plot_from_df(titolo, df):

    ##### print the graph for each cat, for each metric
    metrics = ['clicks_tracks', 'ndcg_tracks', 'precision_tracks']
    for prog in range(1, 11):
        vals = df[df.cat == str(prog)]

        for metric in metrics:
            if len(vals[metric].values) > 0:
                to_disp = [float(x) for x in vals[metric].values.ravel()]

                __display_from_array(df=df,
                                     array_to_display=to_disp, titolo=titolo + " cat" + str(prog),
                                     metric=metric, category_num=prog, x_ticks=vals['vars_str'].values,
                                     show=True, save=False)


def __display_from_array(df, array_to_display, titolo, metric, category_num, x_ticks, save, show, pdf_file=None):

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }


    x = np.arange(len(array_to_display))
    y = np.array(array_to_display)

    fig = plt.figure(dpi=150)

    to_add_title = "("+ list(x_ticks)[0].split('=')[0] + ")"

    xticks = [ x.replace(",","\n") for x in x_ticks]

    plt.title(titolo+to_add_title)
    ax = fig.add_subplot(111)


    if metric == "rp":
        metric_name = "R-Precision"
    elif metric == "click":
        metric_name = "CLICKS"
    elif metric == "ndcg":
        metric_name = "NDCG"
    else:
        raise Exception("metric")

    ax.set_ylim(min(array_to_display) * 0.97, max(array_to_display) * 1.03)

    plt.plot(x, y)
    for i, j in zip(x, y):
        ax.annotate('%.5f' % j, xy=(i, j), rotation=45)

    plt.xticks(np.arange(len(xticks)), list(xticks), fontsize=6, rotation=12)
    plt.ylabel(metric_name, fontdict=font)
    plt.grid()

    if pdf_file is not None:
        plt.savefig(pdf_file, format='pdf')

    if save:
        plt.savefig(ROOT_DIR+"/results/img/" + titolo + "-" + metric + "-cat=" + category_num,
                    dpi=150, )
    if show:
        plt.show()
    plt.close(fig)


def __display_from_three_arrays(rp_array, click_array, ndcg_array, titolo, category_num, x_ticks,y_label,
                                save, show, x_ticks_filtered=None, pdf_file=None):

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

    x = x_ticks
    y1 = np.array(rp_array)
    y2 = np.array(click_array)
    y3 = np.array(ndcg_array)


    fig = plt.figure()
    host = fig.add_subplot(111)

    # ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(111)
    # ax3 = fig.add_subplot(111)
    # plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    rp_min = np.min(rp_array)
    rp_max = np.max(rp_array)
    click_min = np.min(click_array)
    click_max = np.max(click_array)
    ndcg_min = np.min(ndcg_array)
    ndcg_max = np.max(ndcg_array)
    # host.set_xlim(min(rp_array),max(rp_array) )

    host.set_xlim(min(x_ticks), max(x_ticks) )
    host.set_ylim(rp_min * 0.97,rp_max * 1.03)
    par1.set_ylim(click_min * 0.99, click_max * 1.01)
    par2.set_ylim(ndcg_min * 0.97, ndcg_max * 1.03)

    host.set_xlabel("Variabile "+str(y_label))
    host.set_ylabel("R-Precision",fontdict=font)
    par1.set_ylabel("Clicks", fontdict=font)
    par2.set_ylabel("NDCG", fontdict=font)

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.9)
    # "%.2f kg = %.2f lb = %.2f gal = %.2f l" % (var1, var2, var3, var4
    p1, = host.plot(x, y1, color=color1, label="R-Prec, min:%.5f, max:%.5f" % (rp_min, rp_max) )
    p2, = par1.plot(x, y2, color=color2, label="Clicks, min:%.4f, max:%.4f" % (click_min,click_max))
    p3, = par2.plot(x, y3, color=color3, label="NDCG, min:%.5f, max:%.5f" % (ndcg_min,ndcg_max))

    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')
    par2.spines['right'].set_position(('outward', 60))

    par2.xaxis.get_view_interval()
    # if x_ticks_filtered is None:
    # par2.xaxis.set_ticks(x_ticks)
    # else:
    #     par2.xaxis.set_ticklabels(x_ticks_filtered)

    par2.yaxis.set_ticks_position('right')

    # plt.xticks(np.arange(len(xticks)), list(xticks), fontsize=6, rotation=30)

    # par3.set_ylim(min(ndcg_array) * 0.97, max(ndcg_array) * 1.03)

    # host.legend()

    # host.axis["left"].label.set_color(p1.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())
    # par2.axis["right"].label.set_color(p3.get_color())

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    plt.title(titolo)

    if pdf_file is not None:
        plt.savefig(pdf_file, format='pdf',  bbox_inches='tight')

    if save:
        plt.savefig(ROOT_DIR+"/results/img/" + titolo + "-ALLMETRICS-cat=" + category_num, dpi=150, bbox_inches='tight' )
    if show:
        plt.show( bbox_inches='tight')
    plt.close(fig)

    # plt.plot(x, y)
    # for i, j in zip(x, y):
    #     ax.annotate('%.5f' % j, xy=(i, j), rotation=45)









if __name__ == '__main__':




    plot_graph_from_simones_file("prova3_full.csv", pdf_name="BECCATELO", all_in_one=True)


    # plot_results_by_name(path_csvs='../results/test2/',
    #                     algorithm_name="tversky", variable_name='alpha',
    #                     knn='100', shrink='0', beta='1',
    #                     variable_position=2,
    #                     show=True, save=True,
    #                     verbose=True)



