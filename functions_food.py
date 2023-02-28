import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os

#############################################################################
# 1) Sum up the total animal and plant product production for given year
def animal_or_plant(dat, year = 'Y2013', plot_color = 'g'):
    tot_prod = dat.groupby(by='Animal_or_plant_name')[year].sum().reset_index()

    plt.style.use("seaborn-white")
    plt.bar(tot_prod['Animal_or_plant_name'], tot_prod[year], color = plot_color)
    plt.title(f'Total global animal vs plant production in {year[1:]}')
    plt.ylabel('Total annual production (1000 tonnes)')
    plt.xlabel('Category')
    plt.xticks(rotation = 60)
    plt.show()


##############################################################################
# 2) Check distribution of the features

def feature_distribution(dat, transformation = np.sqrt, tr = 'sqrt', colr = 'g', trans_dat = 'yes', folder = 'graphs',\
 features = ['Y2012', 'Y2011', 'Y2006', 'Y2010', 'Y2000', 'Y2008', 'Y1997', 'Y2009', 'Y1995', 'Y2005', 
'Y2002', 'Y2003', 'Y1993', 'Y1992', 'Y2001', 'Y1999', 'Item Code', 'Animal_or_plant_code', 'Y1994']):
    # create graphs folder in working directory
    if not os.path.exists(folder):
        os.makedirs(folder)

    for x in features:
        if trans_dat == 'yes':
            
            #log transformation
            if tr == 'log':
                dat2 = transformation(dat[x])
                plt.style.use("seaborn-white")
                plt.hist(dat2[np.isfinite(dat2)].values, color = colr)
                plt.title(f'Distribution of data for column: {x}')
                plt.ylabel('Count')
                plt.xlabel('Log of production total')
                plt.savefig(f'{folder}/Feature_distribution_{tr}_{x}.png')
                plt.close()
            else:
                dat1 = dat
            # transform data
                dat2 = transformation(dat[x])
                plt.style.use("seaborn-white")
                plt.hist(dat2, color = colr)
                plt.title(f'Distribution of data for column: {x}')
                plt.ylabel('Count')
                plt.xlabel(f'{tr} of production total')
                plt.savefig(f'{folder}/Feature_distribution_{tr}_{x}.png')
                plt.close()
            
            #plot untransformed data
        elif trans_dat == 'no':
            plt.style.use("seaborn-white")
            dat[x].plot(kind='hist', edgecolor='black', color = colr)  
            plt.title(f'Distribution of data for column: {x}')
            plt.ylabel('Count')
            plt.xlabel('Production total (1000 tonnes)')                  
            plt.savefig(f'{folder}/Feature_distribution_original_{x}.png')
            plt.close()
            # if missing required input
        else:
            print(f'Please provide a value for trans_dat = yes or no. (Yes to apply a transformation to the data.)')


# ref 6

#############################################################################
# 3) Forward feature selection
# (from lab 4s)

def feature_selection_func(features, X_train, y_train, model_type = LinearRegression(), show_step = True):

    show_steps = show_step
    included = []

    # keep track of model and parameters
    best = {'feature': '', 'r2': 0, 'a_r2': 0}

    # create a model object to hold the modelling parameters
    model = model_type # create a model for Linear Regression
    # get the number of cases in the training data

    n = X_train.shape[0]

    while True:
        changed = False
        if show_steps:
            print('')


    # list the features to be evaluated
        excluded = list(set(features.columns) - set(included))

        if show_steps:
            print('(Step) Excluded = %s' % ', '.join(excluded))

        # for each remaining feature to be evaluated
        for new_column in excluded:

            if show_steps:
                print('(Step) Trying %s...' % new_column)
                print('(Step) - Features = %s' % ', '.join(included + [new_column]))

            # fit the model with the Training data
            fit = model.fit(X_train[included + [new_column]], y_train) # fit a model; consider which predictors should be included

            # calculate the score (R^2 for Regression)
            r2 = fit.score(X_train[included + [new_column]], y_train) # calculate the score
            # number of predictors in this model
            k = len(included + [new_column])
            # calculate the adjusted R^2
            adjusted_r2 = 1 -(((1-r2)*(n-1))/(n-k-1)) # calculate the Adjusted R^2

            if show_steps:
                print('(Step) - Adjusted R^2: This = %.3f; Best = %.3f' %
                    (adjusted_r2, best['a_r2']))

            # if model improves
            if adjusted_r2 > best['a_r2']:
                # record new parameters
                best = {'feature': new_column, 'r2': r2, 'a_r2': adjusted_r2}
                # flag that found a better model
                changed = True
                if show_steps:
                    print('(Step) - New Best!   : Feature = %s; R^2 = %.3f; Adjusted R^2 = %.3f' %
                        (best['feature'], best['r2'], best['a_r2']))

        # END for

        # if found a better model after testing all remaining features
        if changed:
            # update control details
            included.append(best['feature'])
            excluded = list(set(excluded) - set(best['feature']))
            print('Added feature %-4s with R^2 = %.3f and adjusted R^2 = %.3f' %
                (best['feature'], best['r2'], best['a_r2']))
        else:
            # terminate if no better model
            break

        print('')
        print('Resulting features:')
        print(', '.join(included))

# ref 9