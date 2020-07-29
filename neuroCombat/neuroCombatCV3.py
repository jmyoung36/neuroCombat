#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:21:38 2019

@author: jonyoung
"""

from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
from neuroCombat import make_design_matrix, standardize_across_features, fit_LS_model_and_find_priors, find_parametric_adjustments, adjust_data_final
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy.linalg as la

def fit_transform_neuroCombat(data,
                covars,
                batch_col,
                discrete_cols=None,
                continuous_cols=None):
    """
    Run ComBat to correct for batch effects in neuroimaging data in training data
    Also return correction parameters so they can be applied to testing data with apply_neuroCombat_model

    Arguments
    ---------
    data : a pandas data frame or numpy array
        neuroimaging data to correct with shape = (samples, features)
        e.g. cortical thickness measurements, image voxels, etc

    covars : a pandas data frame w/ shape = (samples, features)
        demographic/phenotypic/behavioral/batch data 
        
    batch_col : string
        - batch effect variable
        - e.g. scan site

    discrete_cols : string or list of strings
        - variables which are categorical that you want to predict
        - e.g. binary depression or no depression

    continuous_cols : string or list of strings
        - variables which are continous that you want to predict
        - e.g. depression sub-scores

    Returns
    -------
    - A numpy array with the same shape as `data` which has now been ComBat-corrected
    - LS_dict - dictionary of correction parameters learned from the training data and covariates
    """
    ##############################
    ### CLEANING UP INPUT DATA ###
    ##############################
    if not isinstance(covars, pd.DataFrame):
        raise ValueError('covars must be pandas datafrmae -> try: covars = pandas.DataFrame(covars)')

    if not isinstance(discrete_cols, (list,tuple)):
        if discrete_cols is None:
            discrete_cols = []
        else:
            discrete_cols = [discrete_cols]
    if not isinstance(continuous_cols, (list,tuple)):
        if continuous_cols is None:
            continuous_cols = []
        else:
            continuous_cols = [continuous_cols]

    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 
    for i in range(covars.shape[-1]):
        try:
            covars[:,i] = covars[:,i].astype('float32')
        except:
            pass

    if isinstance(data, pd.DataFrame):
        data = np.array(data, dtype='float32')
    data = data.T # transpose data to make it (features, samples)... a weird genetics convention..

    ##############################

    # get column indices for relevant variables
    batch_col = np.where(covar_labels==batch_col)[0][0]
    cat_cols = [np.where(covar_labels==c_var)[0][0] for c_var in discrete_cols]
    num_cols = [np.where(covar_labels==n_var)[0][0] for n_var in continuous_cols]

    # conver batch col to integer
    covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]
    # create dictionary that stores batch info
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    }

    # create design matrix
    print('Creating design matrix..')
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols)
    
    # standardize data across features
    print('Standardizing data across features..')
    s_data, s_mean, v_pool, grand_mean, B_hat = standardize_across_features_train(data, design, info_dict)
    
    # fit L/S models and find priors
    print('Fitting L/S model and finding priors..')
    LS_dict = fit_LS_model_and_find_priors(s_data, design, info_dict)

    # find parametric adjustments
    print('Finding parametric adjustments..')
    gamma_star, delta_star = find_parametric_adjustments(s_data, LS_dict, info_dict)

    # adjust data
    print('Final adjustment of data..')
    bayes_data = adjust_data_final(s_data, design, gamma_star, delta_star, 
                                                s_mean, v_pool, info_dict)

    bayes_data = np.array(bayes_data)
    
#    return all this lot: design, var_pooled, B_hat, grand_mean,
#        gamma_star, delta_star, info_dict (a neuroCombat invention),
#        gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior
#    in a single model structure
    model = {'design': design,
             'var_pooled':v_pool, 'B_hat':B_hat, 'grand_mean': grand_mean,
             'gamma_star': gamma_star, 'delta_star': delta_star, 'info_dict': info_dict,
             'gamma_hat': LS_dict['gamma_hat'], 'delta_hat': np.array(LS_dict['delta_hat']),
             'gamma_bar': LS_dict['gamma_bar'], 't2': LS_dict['t2'],
             'a_prior': LS_dict['a_prior'], 'b_prior': LS_dict['b_prior']}

    return bayes_data.T, model

def apply_neuroCombat_model(data,
                covars,
                model,
                batch_col,
                discrete_cols=None,
                continuous_cols=None):
    """
    Run ComBat to correct for batch effects in neuroimaging data on testing data
    Take correction parameters in LS_dict produced by fit_transform_neuroCombat

    Arguments
    ---------
    data : a pandas data frame or numpy array
        neuroimaging data to correct with shape = (samples, features)
        e.g. cortical thickness measurements, image voxels, etc

    covars : a pandas data frame w/ shape = (samples, features)
        demographic/phenotypic/behavioral/batch data 
        
        model : a dictionary of model parameters
        the output of a call to train_neuroCombat_model()
        
   batch_col : string
        - batch effect variable
        - e.g. scan site

   discrete_cols : string or list of strings
        - variables which are categorical that you want to predict
        - e.g. binary depression or no depression

   continuous_cols : string or list of strings
        - variables which are continous that you want to predict
        - e.g. depression sub-scores

    Returns
    -------
    - A numpy array with the same shape as `data` which has now been ComBat-corrected
    """
    ##############################
    ### CLEANING UP INPUT DATA ###
    ##############################
    if not isinstance(covars, pd.DataFrame):
        raise ValueError('covars must be pandas datafrmae -> try: covars = pandas.DataFrame(covars)')

    if not isinstance(discrete_cols, (list,tuple)):
        if discrete_cols is None:
            discrete_cols = []
        else:
            discrete_cols = [discrete_cols]
    if not isinstance(continuous_cols, (list,tuple)):
        if continuous_cols is None:
            continuous_cols = []
        else:
            continuous_cols = [continuous_cols]

    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 
    for i in range(covars.shape[-1]):
        try:
            covars[:,i] = covars[:,i].astype('float32')
        except:
            pass

    if isinstance(data, pd.DataFrame):
        data = np.array(data, dtype='float32')
    data = data.T # transpose data to make it (features, samples)... a weird genetics convention..

    ##############################

    # get column indices for relevant variables
    batch_col = np.where(covar_labels==batch_col)[0][0]
    cat_cols = [np.where(covar_labels==c_var)[0][0] for c_var in discrete_cols]
    num_cols = [np.where(covar_labels==n_var)[0][0] for n_var in continuous_cols]

    # conver batch col to integer
    covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]
    # create dictionary that stores batch info
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    }

    # create design matrix
    # ignore any cat_cols/discrete_cols in test sample
    print('Creating design matrix..')
    design = make_design_matrix(covars, batch_col, [], [])
    
#    # standardize data across features
#    print('Standardizing data across features..')
#    s_data, s_mean, v_pool = standardize_across_features_test(data, design, info_dict)

    # standardize data across features
    # modified to use parameters from model from training data
    print('Standardizing data across features..')
    s_data, stand_mean, var_pooled = standardize_across_features_test(data, design, info_dict, model)

    # adjust data
    print('Final adjustment of data..')
    bayes_data = adjust_data_final(s_data, design, model['gamma_star'],  model['delta_star'], 
                                                stand_mean, var_pooled, info_dict)

    bayes_data = np.array(bayes_data)

    return bayes_data.T

def standardize_across_features_train(X, design, info_dict):
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), X.T)
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled, grand_mean, B_hat


# modofied from standard standardize_across_features to accept v_pool from training data
def standardize_across_features_test(X, design, info_dict, model):
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']
    
    # remove adjustments for continuous/discrete cols in test
    # ie any after n_batch
    B_hat = model['B_hat']
    B_hat = B_hat[:n_batch]
    grand_mean = model['grand_mean']
    var_pooled = model['var_pooled']

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled

# utility to do CV that is compatible with neuroCombat_CV
# like ShuffleSplit
# but ensures there is are at least three of each group in the TRAINING set
# and at least one of each in the TEST set
# before randomly splitting remaining examples
# so must have at least three of each group
def ShuffleSplitFixed(X, y, group, test_fraction, n_splits) :
    
    n_rows = np.shape(X)[0]
    n_groups = len(set(group))
    train_size = int(np.ceil((1 - test_fraction) * n_rows))
    test_size = n_rows - train_size
    batches_counts = np.unique(group, return_counts = True)
    n_batches = len(batches_counts[0])
    
    # check inputs are the correct size
    if (not n_rows == np.shape(y)[0]) or (not n_rows == np.shape(group)[0]) :

        print ('X, y and group must all have same number of rows!')
        return -1
    
    # check we can have two of each group in training 
    elif train_size < 3 * n_groups :
        
        print ('Size of training set must be at least equal to twice the number of groups!')
        return -1
        
    # check if we can have one of each group in testing
    elif test_size < n_groups :
            
        print ('Size of training set must be at least equal to number of groups!')
        return -1
    
    # check we have at least 3 of EACH group in the whole dataset    
    elif any(batches_counts[1] < 4) :
        
        print ('All groups must contain at least 4 instances!')
        print ('Remove the following group(s) or add more instances to them:')
        for i, group in enumerate(batches_counts[0]) :
            
            if batches_counts[1][i] < 4 :
                
                print ('Group ' + str(group))
                
        return -1
    
    # everything is OK - make the split!
    else :
        
        # create lists to hold all sets of train inds and test inds
        test_inds_all = []
        train_inds_all = []
        
        for i in range(n_splits) :
        
            # create a list of train and test inds for each group
            test_inds = []
            train_inds = []
            for batch in batches_counts[0] :
                
                # find indices of this batch
                batch_inds = np.nonzero(group == batch)[0]

                # shuffle and allocate first two to train and third to test
                np.random.shuffle(batch_inds)
                train_inds = train_inds + list(batch_inds[:3])
                test_inds.append(batch_inds[3])

            # at this stage we have allocated the minimum numbers to the train and testing sets.
            # now split the rest
            # start by generating a list of unallocated indices
            all_inds = range(n_rows)
            unallocated_inds = [ind for ind in all_inds if (not ind in train_inds) and (not ind in test_inds)]
            unallocated_inds = np.array(unallocated_inds)
            
            # use StratifiedShuffleSplit to break unallocated indices
            # start by calculating the size of the required test set
            unallocated_test_size = test_size - len(test_inds)
            
            # make dummy data for sss
            X_dummy = X[unallocated_inds, :]
            
            # get groups of unallocated instances
            group_unallocated = group[unallocated_inds]
                        
            # if possible stratfy by group again
            unique, unique_counts = np.unique(group_unallocated, return_counts=True)            
            if unallocated_test_size >= n_batches and np.min(unique_counts) > 1 :
                
                sss = StratifiedShuffleSplit(n_splits=1, test_size=unallocated_test_size)
                for train_index, test_index in sss.split(X_dummy, group_unallocated) :
                    
                    train_inds_unllocated = list(unallocated_inds[np.array(train_index)])
                    test_inds_unallocated = list(unallocated_inds[np.array(test_index)])
                
                    train_inds = train_inds + train_inds_unllocated
                    test_inds = test_inds + test_inds_unallocated
                    
            else :
                
                # do split of unallocated inds
                # if unallocated_test_size is zero, everything else goes in training
                if unallocated_test_size == 0 :
                
                    train_inds = train_inds + list(unallocated_inds)
                    
                else :
                    
                    ss = ShuffleSplit(n_splits=1, test_size=unallocated_test_size)
                    for train_index, test_index in ss.split(X_dummy) :
                    
                        train_inds_unllocated = list(unallocated_inds[np.array(train_index)])
                        test_inds_unallocated = list(unallocated_inds[np.array(test_index)])
                        
                        train_inds = train_inds + train_inds_unllocated
                        test_inds = test_inds + test_inds_unallocated
           
            test_inds_all.append(np.array(test_inds))
            train_inds_all.append(np.array(train_inds))
            
        return train_inds_all, test_inds_all, test_size