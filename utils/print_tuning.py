import os
import pandas as pd
from utils.plot_results_csv import plot_graph_from_simones_file

class TunePrint:
    def __init__(self, filename, full=True, mean=True, overwrite=False, verbose=True):

        self.filename_full=os.path.join(filename,filename+'_full.csv')

        self.filename_mean=os.path.join(filename,filename+'_mean.csv')

        self.dataframe_full = pd.DataFrame()
        self.dataframe_mean = pd.DataFrame()
        self.mean = mean
        self.full = full
        self.verbose = verbose

        if not os.path.exists(filename):
            os.makedirs(filename)
        

        if mean and os.path.isfile(self.filename_mean) and not overwrite:
            raise AssertionError('ERROR, %s already exist and overwrite flag is set to False'%self.filename_mean)

        elif mean and os.path.isfile(self.filename_mean) and overwrite:
            if self.verbose:print("WARNING, overwriting file",self.filename_mean)
            os.remove(self.filename_mean)

        if full and os.path.isfile(self.filename_full) and not overwrite:
            raise AssertionError('ERROR, %s already exist and overwrite flag is set to False'%self.filename_full)

        elif full and os.path.isfile(self.filename_full) and overwrite:
            if self.verbose:print("WARNING, overwriting file",self.filename_full)
            os.remove(self.filename_full)


    def print_full_values(self, dict_val, dataframe, description=None, df_version = False):
        assert(self.full==True)

        if df_version:
            # se vuoi la versione col df che è un po' piu leggibile  da file ma devo testarla con un main
            col_names = [ x for x in dict_val]
            values = [ dict_val[x] for x in dict_val]
            col_names.extend( ['clicks_tracks', 'ndcg_tracks', 'precision_tracks'])
            values.extend( dataframe[['clicks_tracks', 'ndcg_tracks', 'precision_tracks']].values)

            to_append = pd.DataFrame([values], columns=col_names)
            self.dataframe_full = self.dataframe_full.append( to_append )

            with open(self.filename_full, 'a') as file:
                file.write("# "+description+'\n')
                to_append.to_csv(file, mode='a', header=False)
            file.close()

        else :

            file = open(self.filename_full, 'a+')
            if description is not None:
                file.write('-----' + description + '\n')

            file.write("###\n")
            # write the values
            file.write( str(dict_val).replace("{","").replace("}","").replace(":","=").replace("'","")+'\n' )

            #for each category, cat = i+1
            for i in range(1,11):
                values = dataframe.loc["cat"+str(i)]
                file.write("##"+str(i)+'\n')
                file.write(str(values['precision_tracks']) + "," +
                           str(values['ndcg_tracks']) + "," +
                           str(values['clicks_tracks']) + "\n")

            file.close()

        if self.verbose: print(dataframe[['clicks_tracks', 'ndcg_tracks', 'precision_tracks']])


    def print_mean_values(self, description, mean):
        assert(self.mean==True)
        print(mean)
        with open(self.filename_mean,'a+') as file:
                        file.write('%s\nP = %1.4f, NDCG = %1.4f, CLICK = %1.4f\n'%(description,mean[0],mean[1],mean[2]))


    def print_values(self, description, values, dataframe):
        pass


    def make_pdf_full(self):
        plot_graph_from_simones_file(self.filename_full,self.filename_full.split(".")[0])

    def make_pdf_mean(self):
        pass

def pdf_from_file(filename):
    plot_graph_from_simones_file(filename, filename.split(".")[0])