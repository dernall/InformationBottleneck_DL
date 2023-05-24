import matplotlib.pyplot as plt
import numpy as np
import random
import os
from scipy.stats import norm
import scipy
from utils_f import load_code
import os    



def log2_stable(value):
    """ Return the base 2 logarithm. Numerically stable is ensured by catching special cases, namely 0. """
    if np.any(value <= 0):
        if np.isscalar(value):
            return -1e6
        result = np.empty_like(value)
        result[value > 0] = np.log2(value[value > 0])
        result[value <= 0] = -1e6
        return result
    return np.log2(value)


def mutual_information(input_pdf):
    """ Return the Mutual Information for a given base an joint distribution sum over joint matrix over the rows
    to determine p_x.
    Args:
    input_pdf = a 2D array containing the joint probabilities. For p(x,y) x is fixed for a particular column and y is
                fixed in one row.
    """
    p_x = input_pdf.sum(0, keepdims=True)
    # sum over joint matrix over the columns to determine p_y
    p_y = input_pdf.sum(1, keepdims=True)
    MI = (input_pdf * log2_stable(input_pdf / (p_y.dot(p_x) + 1e-31))).sum()
    return MI


def kl_divergence(pdf1, pdf2):
    """ Return the Kullback-Leibler Divergence for two input PDFs. For use in IB algorithm.
    Note:
        The KL-divergence is not a real metric, the order of the input matters.
    Args:
    pdf1 =  a 2D array containing the joint probabilities. For p(x|y) x is fixed for a particular column and y is
            fixed in one row.
    pdf2 =  a 2D array containing the joint probabilities. For p(x|t) t is fixed for a particular column and x is
            fixed in one row.
    Note:
        It is also possible to input only an 1D array, which will be extended automatically using broadcasting.
        If the first pdf1 is a 1D arrays it is extended with pdf1[np.newaxis,:] to turn it in a row vector.
    """
    if pdf1.ndim == 1:
        pdf1 = pdf1[np.newaxis, :]
    KL_vec = (pdf1 * log2_stable(pdf1 / (pdf2 + 1e-31))).sum(1)
    return KL_vec




def js_divergence(pdf1, pdf2, pi1, pi2):
    """Return the Jenson-Shannen Divergence for two input PDFs and given pi1 and pi2.
    Note:
        The JS is the symmetrized and smoothed version of the KL-divergence.
    Args:
    pdf1 = a 2D array containing joint probabilities.
    pdf2 = a 2D array containing joint probabilities.
    pi1 = weighting factor
    pi2 = weighting factor
    """
    # catch special case that pi1 is a vector appears for sIB algorihtms
    if False in (np.isscalar(pi1), np.isscalar(pi2)):
        p_tilde_mat = pi1[:, np.newaxis] * pdf1 + pi2[:, np.newaxis] * pdf2
    else:
        p_tilde_mat = pi1 * pdf1 + pi2 * pdf2
    JS_vec = pi1 * kl_divergence(pdf1, p_tilde_mat) + pi2 * kl_divergence(pdf2, p_tilde_mat)
    return JS_vec


# ## Running Information Bottleneck Algorithms

# In this section, we will use three exemplary information bottleneck algorithm to show the syntax of the respective algorithms and how to access the lookup table $p(t|y)$ and the cluster meaning $p(x|t)$.

# In[5]:


import sys

class GenericIB:
    """Common base class for all Information Bottleneck classes
    Args:
    input parameter
        p_x_y                   input joint pdf, where x is the number of columns and y the number of rows
    IB related parameters
        cardinality_T
        beta
        eps
        nror
    mutual information
        MI_XT                   mutual information of output I(X;T)
        MI_XY                   mutual information of input I(X;Y)
    output PDF_s
        p_t_given_y
        p_x_given_t
        p_t
    """

    def __init__(self, p_x_y_, card_T_, beta_, eps_ , nror_):
        # initialize parameters
        self.p_x_y = p_x_y_
        self.cardinality_T = card_T_
        self.beta = beta_
        self.eps = eps_
        self.nror = nror_

        # initialize unused parameters
        self.MI_XT = 1
        self.MI_XY = 1
        self.p_t_given_y = np.zeros((self.p_x_y.shape[0], self.cardinality_T))
        self.p_x_given_t = np.zeros((self.cardinality_T, self.p_x_y.shape[1]))
        self.p_t = np.zeros((1,self.cardinality_T))
        self.name = 'GenericClass'

        if card_T_ >= self.p_x_y.shape[0]:
            raise RuntimeError('The number of desired clusters is larger/equal than the input cardinality |T|>=|Y| !!')

    def calc_merger_cost(self):
        """Return the merger cost for putting one event in a cluster.
        Args:
            p_t: is a 1 x card_T array
            p_x_given_t: is a card_X x card_T array
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element
        p_t_bar = self.p_t[ -1] + self.p_t[:- 1]

        pi1 = self.p_t[-1] / p_t_bar
        pi2 = self.p_t[:-1] / p_t_bar

        cost_vec = p_t_bar * (js_divergence(self.p_x_given_t[-1, :], self.p_x_given_t[:-1, :], pi1, pi2) -
                    (pi1 * np.log2(pi1)+pi2 * np.log2(pi2)) / self.beta)

        return cost_vec

    def run_IB_algo(self):
        """only template that will be used by the specific implementations later."""
        pass

    def return_results(self):
        """Return all parameters generate by the Information Bottleneck as dictionary
        Return:
            :return p_t_given_y:
            :return p_x_given_t:
            :return p_t:
            :return MI_XT:
            :return MI_XY:
        Note: These are the strings of the dictionary elements.
        """
        return {'p_t_given_y': self.p_t_given_y,
                'p_x_given_t': self.p_x_given_t,
                'p_t': self.p_t,
                'MI_XT': self.MI_XT,
                'MI_XY': self.MI_XY }

    def display_MIs(self,short=False):
        """Return the Mutual information for the input pdf and after applying the IB as well as the ratio of both in a
        graphical way.
        Args:
            None
        Return:
            None
        """
        if short:
            print('MI_XT_s= ', str(self.MI_XT))
        else:
            print('----- Mutual Information Comp --- ')
            print('----- ', self.name, ' ------ ')
            print('MI_XT_s= ', str(self.MI_XT))
            print('MI_XY_s= ', str(self.MI_XY))
            print('ratio= ', str(self.MI_XT / self.MI_XY))

    def printProgress(self, iteration, total, prefix='', suffix='', decimals=2, barLength=70):
        """
        Call in a loop to create terminal progress bar
        Args:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : number of decimals in percent complete (Int)
            barLength   - Optional  : character length of bar (Int)
        """
        filledLength = int(round(barLength * (iteration + 1) / float(total)))
        percents = round(100.00 * ((iteration + 1) / float(total)), decimals)
        bar = '#' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r')
        sys.stdout.write('%s [%s] %s%s %s / %s   %s \r' % (prefix, bar, percents, '%', iteration + 1, total, suffix)),
        sys.stdout.flush()
        if iteration+1 == total:
            print("\n")

    def get_results(self):
        return self.p_t_given_y, self.p_x_given_t, self.p_t

    def get_mutual_inf(self):
        return self.MI_XT, self.MI_XY


# ### Sequential Symmetric Information Bottleneck Algorithm [LB18]

# In[6]:


class symmetric_sIB(GenericIB):
    """This class realizes the calculation of the modified sequential Information Bottleneck algorithm that outputs
        symmetric clusters.
      Description:
        The class implements the algorithm for the creation of the Trellis
        message mappings from the paper. It is an extended version of the
        modified sequential IB algorithm that additionally enforces symmetry of
        the clusters.
      Args:
      input parameter
          p_x_y                     input joint pdf, where x is the number of columns and y the number of rows
      IB related parameters
          cardinality_T
          beta                      is set to Inf for the sIB
          eps                       not used in this algorithm set to []
          nror                      number of runs
      Return:
      mutual information
          MI_XT                     mutual information of output I(X;T)
          MI_XY                     mutual information of input I(X;Y)
      output PDF_s
          p_t_given_y
          p_x_given_t
          p_t
      Note: The values can be accessed using the appropriate return functions of the class
        self.return_results         return all values
        self.display_MI             return the MIs graphically as well as percentage of maintained mutual information
      """

    def __init__(self, p_x_y_, card_T_, nror_):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], nror_)
        self.name = 'symmetric sIB'

    def run_IB_algo(self):
        if self.cardinality_T == 2:
            self.cardinality_Y = self.p_x_y.shape[0]
            self.p_t_given_y = np.zeros((self.cardinality_Y, 2))
            self.p_t_given_y[0:int(self.cardinality_Y / 2) , 1] = 1
            self.p_t_given_y[int(self.cardinality_Y / 2):, 0] = 1
            self.p_y=self.p_x_y.sum(axis=1)

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :self.cardinality_T] * self.p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:self.cardinality_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :self.cardinality_T].T,
                                                                            self.p_x_y)
        else:
            self.symmetric_sIB_algo()

    def calc_merger_cost(self, border_between_clusters, cur_card_T_):
        """Return the merger cost for putting one event in a cluster. Since this a modified version of the sIB only two
            clusters have to be tested. Which constrains the calculation to two comparisons.
            Args:
                :param border_between_clusters: denotes the current border between two adjacent cluster, optimized in this step
                :param cur_card_T_: The current cardinality T, meaning the cluster size, which is increased during the
                                    algorithm due to temporary clusters
            Return
                :return cost_vec:
            """

        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element
        bbc = border_between_clusters
        cur_card_T = cur_card_T_-2
        p_t_bar = self.p_t[cur_card_T] + self.p_t[[bbc, bbc+1]]

        pi1 = self.p_t[cur_card_T] / p_t_bar
        pi2 = self.p_t[[bbc, bbc+1]] / p_t_bar

        cost_vec = p_t_bar * (js_divergence(self.p_x_given_t[cur_card_T, :], self.p_x_given_t[[bbc, bbc+1], :], pi1, pi2)
                              -(pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)

        return cost_vec

    def symmetric_sIB_algo(self):
        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X=p_x.shape[0]
        cardinality_Y=p_y.shape[0]

        cur_card_T = self.cardinality_T

        # Initialization

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_YT = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.cardinality_T , self.nror))
        p_t_mats = np.zeros((1, self.cardinality_T, self.nror))
        p_x_given_t_mats = np.zeros((self.cardinality_T, cardinality_X, self.nror))
        all_needed = 0
        # run for-loop for each number of run
        for run in range(0, self.nror):
            self.printProgress(run, self.nror,prefix='Run:')

            self.p_t_given_y = np.zeros((cardinality_Y, self.cardinality_T + 2))

            # Initialization of p_t_given_y
            # Use dirichlet distribution to sample a valid density
            # the ones vector indicate that The expected value for each cluster is 1/(cardinaltiy_T)
            # Multiplying with constant changes variance. 1 means very high variance creating asymmetric clusters.
            # In the end +1 is needed to ensure that
            alpha = np.ones(int(self.cardinality_T / 2)) * 1
            border_vec = np.ones(alpha.shape[0]) * cardinality_Y / 2
            while border_vec[:-1].cumsum().max() >= cardinality_Y / 2:
                border_vec = np.floor(np.random.dirichlet(alpha, 1).transpose() * (cardinality_Y / 2))
                border_vec[border_vec == 0] = 1

            border_vec = np.hstack([border_vec[:-1].cumsum(), cardinality_Y / 2]).astype(int)

            a = 0
            for t in range(0, int(self.cardinality_T / 2)):
                self.p_t_given_y[a:border_vec[t], t] = 1
                self.p_t_given_y[cardinality_Y - border_vec[t] :cardinality_Y - a, self.cardinality_T - t -1] = 1
                a = border_vec[t]


            # Processing
            init_mat = self.p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.cardinality_T + 2))

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)

                all_needed +=1

                last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 1, 0])
                partner_last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 0, 1])

                init_mat = np.copy(self.p_t_given_y)

                for border_between_clusters in range(0,int(self.cardinality_T/2) - 1):
                    done_left_to_right = False
                    done_right_to_left = False

                    while not done_left_to_right:
                        done_left_to_right = True
                        all_needed += 1
                        # find last element in the cluster
                        # this is a trick here because argmax returns first hit so we flip the array first.
                        last_elem = self.p_t_given_y.shape[0] - np.argmax(self.p_t_given_y[::-1, border_between_clusters] > 0) - 1

                        old_cluster = border_between_clusters

                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[last_elem, :] = last_cluster_vec
                            self.p_t_given_y[-(last_elem+1), :] = partner_last_cluster_vec

                            cur_card_T += 2

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1/(self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[last_elem, border_between_clusters] = 1
                                self.p_t_given_y[cardinality_Y-last_elem-1, cur_card_T-2-border_between_clusters-1] = 1

                            else:
                                self.p_t_given_y[last_elem, border_between_clusters+1] = 1
                                self.p_t_given_y[- (last_elem+1), cur_card_T - 2 - border_between_clusters-2] = 1
                                done_left_to_right = False

                            self.p_t_given_y[-(last_elem+1), -1] = 0
                            self.p_t_given_y[last_elem, - 2] = 0

                            cur_card_T -= 2

                    # check other direction
                    while not done_right_to_left:
                        done_right_to_left = True
                        all_needed += 1
                        # find first element in the cluster
                        first_elem = np.argmax(self.p_t_given_y[:, border_between_clusters + 1] > 0)

                        old_cluster = border_between_clusters + 1
                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[first_elem, :] = last_cluster_vec
                            self.p_t_given_y[cardinality_Y - first_elem - 1, :] = partner_last_cluster_vec

                            cur_card_T += 2

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[first_elem, border_between_clusters] = 1
                                self.p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 2 - border_between_clusters-1] = 1

                                done_right_to_left = False
                            else:
                                self.p_t_given_y[first_elem, border_between_clusters + 1] = 1
                                self.p_t_given_y[cardinality_Y - first_elem-1, cur_card_T - 2 - border_between_clusters-2] = 1

                            self.p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 1] = 0
                            self.p_t_given_y[first_elem, cur_card_T - 2] = 0
                            cur_card_T -= 2

                end_mat = self.p_t_given_y

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

            p_t_given_y_mats[:, :, run] = self.p_t_given_y[:, :self.cardinality_T]
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t[:self.cardinality_T, :]

            p_ty = self.p_t_given_y[:, :self.cardinality_T] * p_y[:, np.newaxis]
            p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]

            I_YT[run] = mutual_information(p_ty)
            I_TX[run] = mutual_information(p_xt)

            ib_fct[run] = I_YT[run] / (-self.beta) + I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]
class lin_sym_sIB(GenericIB):
    """This class realizes the calculation of the modified sequential Information Bottleneck algorithm that outputs
        symmetric clusters and accepts an input Y that is a high dimensional vector with linearized indices.
      Args:
      input parameter
          p_x_y                     input joint pdf, where x is the number of columns and y the number of rows
      IB related parameters
          cardinality_T
          beta                      is set to Inf for the sIB
          eps                       not used in this algorithm set to []
          nror                      number of runs
      Return:
      mutual information
          MI_XT                     mutual information of output I(X;T)
          MI_XY                     mutual information of input I(X;Y)
      output PDF_s
          p_t_given_y
          p_x_given_t
          p_t
      Note: The values can be accessed using the appropriate return functions of the class
        self.return_results         return all values
        self.display_MI             return the MIs graphically as well as percentage of maintained mutual information
      """


    TOL_LLR_IS_THE_SAME = 1e-5

    def __init__(self, p_x_y_, card_T_, nror_):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], nror_)
        self.name = 'linearized symmetric sIB'
        self.p_x_and_tilde_y = []
        self.sort_indices = []
        self.reconstruct_ind = []

    def run_IB_algo(self):
        # LLR preclustering
        self.pre_clustering()

        #rename variables
        p_x_y_initial = self.p_x_y.copy()
        self.p_x_y = self.p_x_and_tilde_y.copy()

        if self.cardinality_T == 2:
            self.cardinality_Y = self.p_x_y.shape[0]
            self.p_t_given_y = np.zeros((self.cardinality_Y, 2))
            self.p_t_given_y[0:int(self.cardinality_Y / 2), 1] = 1
            self.p_t_given_y[int(self.cardinality_Y / 2):, 0] = 1
            self.p_y=self.p_x_y.sum(axis=1)

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :self.cardinality_T] * self.p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:self.cardinality_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :self.cardinality_T].T,                                                                            self.p_x_y)
        else:
            # forced_sym_lin_two_dim_input_sIB
            self.lin_sym_sIB_algo()
            self.p_t_given_y = self.p_t_given_y[:, :-2]

        self.p_x_y = p_x_y_initial

        # blow up
        self.blow_up()

    def pre_clustering(self):
        """The idea of this function is provide an initial preclustering or grouping of combinations resulting in the
        same Log Likelihood Ratio (LLR) this idea was first presented in the paper by J. Lewandowsky "Trellis based Node
        Operations for LDPC Decoders from the Information Bottleneck Method" and is now generalized from applications
        beside Discrete Density Evolution. Basically it uses the input distribution p_x_and_vec_y_lin and calculates a
        group version p_x_and_tilde_y. This new distribution is then processed and final the results are used to blow
        up the compressed density to the original size.
        """

        # Start with calculation of corresponding LLRs.

        corresponding_LLRs = np.log(self.p_x_y[:, 0]/ self.p_x_y[:, 1])

        # the next line is necessary to ensure that all required symmetry conditions are fulfilled.Sometimes otherwise
        # numerical instabilities lead to a "false" sorting result that does not  meet the symmetry requirements.
        corresponding_LLRs[np.abs(corresponding_LLRs) < self.TOL_LLR_IS_THE_SAME] = 0

        self.sort_indices = np.argsort(corresponding_LLRs)

        corresponding_LLRs_sorted = corresponding_LLRs[self.sort_indices]

        # now sort the rows of p_x_and_y by the computed LLRs. Important! use np.sort(array) to return a sorted copy.
        # ndarray.sort would sort the array in place

        p_x_y_sorted = self.p_x_y[self.sort_indices, :]

        # We need to calculate p_y_tilde_and_x by partial summation over p_x_and_vec_y_lin To do so, we first check
        # out how many distinct LLRs exist. Then we find out how often each LLR appears in the sorted LLR vector using
        # the next for loop.

        distinct_LLRs, unique_ind, self.reconstruct_ind, number_of_equivalent_sequent_combinations = np.unique(
                                    np.round(corresponding_LLRs_sorted[:int(corresponding_LLRs.size/2)],
                                             -int(np.log10(self.TOL_LLR_IS_THE_SAME))), True, True, True)

        # due to symmetry we have
        number_of_equivalent_sequent_combinations = np.hstack((number_of_equivalent_sequent_combinations,
                                                               number_of_equivalent_sequent_combinations[::-1]))

        # now we know the cardinality of Y_tilde.

        cardinality_Y_tilde = number_of_equivalent_sequent_combinations.shape[0]

        # and now we can perform the summation of eq. 27 in the paper
        self.p_x_and_tilde_y = np.zeros([int(cardinality_Y_tilde/2), 2])
        upper_summation_bounds_y_vec_lin = np.cumsum(number_of_equivalent_sequent_combinations)
        lower_summation_bounds_y_vec_lin = upper_summation_bounds_y_vec_lin - number_of_equivalent_sequent_combinations

        for u in range(0,int(cardinality_Y_tilde / 2)):
            self.p_x_and_tilde_y[u, :] = np.sum(
                p_x_y_sorted[lower_summation_bounds_y_vec_lin[u]:upper_summation_bounds_y_vec_lin[u], :], axis=0)

        self.p_x_and_tilde_y = np.vstack((self.p_x_and_tilde_y, self.p_x_and_tilde_y[::-1, ::-1]))


    def blow_up(self):
        """This function reverts the compression performed by the pre clustering. This is very easy, because the unique
        function of the numpy framework returns the indices to reconstruct the original array."""

        self.p_t_given_y = self.p_t_given_y[np.hstack((self.reconstruct_ind, self.p_t_given_y.shape[0]-1-self.reconstruct_ind[::-1])), :]
        p_t_given_y_blown = self.p_t_given_y.copy()

        #sort back
        self.p_t_given_y[self.sort_indices, :] = p_t_given_y_blown

        p_y = self.p_x_y.sum(1)

        # calculate p(t)  new
        self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)

        # calculate p(x | t) new
        self.p_x_given_t = 1 / (self.p_t[:, np.newaxis]) * np.dot(self.p_t_given_y.T, self.p_x_y)


    def calc_merger_cost(self, border_between_clusters, cur_card_T_):
        """Return the merger cost for putting one event in a cluster. Since this a modified version of the sIB only two
        clusters have to be tested. Which constrains the calculation to two comparisons.
        Args:
            :param border_between_clusters: denotes the current border between two adjacent cluster, optimized in this step
            :param cur_card_T_: The current cardinality T, meaning the cluster size, which is increased during the
                                algorithm due to temporary clusters
        Return
            :return cost_vec:
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element
        bbc = border_between_clusters
        cur_card_T = cur_card_T_-2
        p_t_bar = self.p_t[cur_card_T] + self.p_t[[bbc, bbc+1]]

        pi1 = self.p_t[cur_card_T] / p_t_bar
        pi2 = self.p_t[[bbc, bbc+1]] / p_t_bar

        cost_vec = p_t_bar * (js_divergence(self.p_x_given_t[cur_card_T, :], self.p_x_given_t[[bbc, bbc+1], :], pi1, pi2)
                              -(pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)

        return cost_vec

    def lin_sym_sIB_algo(self):
        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X=p_x.shape[0]
        cardinality_Y=p_y.shape[0]

        cur_card_T = self.cardinality_T

        # Initialization

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_YT = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.cardinality_T + 2, self.nror))
        p_t_mats = np.zeros((1, self.cardinality_T, self.nror))
        p_x_given_t_mats = np.zeros((self.cardinality_T, cardinality_X, self.nror))

        # run for-loop for each number of run
        for run in range(0, self.nror):
            self.printProgress(run, self.nror, prefix='Run:')

            self.p_t_given_y = np.zeros((cardinality_Y, self.cardinality_T + 2))

            # Use dirichlet distribution to sample a valid density
            # the ones vector indicate that The expected value for each cluster is 1/(cardinaltiy_T)
            # Multiplying with constant changes variance. 1 means very high variance creating asymmetric clusters.
            # In the end +1 is needed to ensure that
            alpha = np.ones(int(self.cardinality_T/2)) * 1
            border_vec = np.ones(alpha.shape[0])*cardinality_Y/2
            while border_vec[:-1].cumsum().max() >= cardinality_Y / 2:
                border_vec = np.floor(np.random.dirichlet(alpha, 1).transpose() * (cardinality_Y / 2))
                border_vec[border_vec == 0] = 1

            border_vec = np.hstack([border_vec[:-1].cumsum(), cardinality_Y / 2]).astype(int)

            a = 0
            for t in range(0, int(self.cardinality_T / 2)):
                self.p_t_given_y[a:border_vec[t], t] = 1
                self.p_t_given_y[cardinality_Y - border_vec[t] :cardinality_Y - a, self.cardinality_T - t - 1] = 1
                a = border_vec[t]

            # Processing
            init_mat = self.p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.cardinality_T + 2))

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)

                last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 1, 0])
                partner_last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 0, 1])

                init_mat = np.copy(self.p_t_given_y)

                for border_between_clusters in range(0,int(self.cardinality_T/2) - 1):
                    done_left_to_right = False
                    done_right_to_left = False

                    while not done_left_to_right:
                        done_left_to_right = True

                        # find last element in the cluster
                        # this is a trick here because argmax returns first hit so we flip the array first.
                        last_elem = self.p_t_given_y.shape[0] - np.argmax(self.p_t_given_y[::-1, border_between_clusters] > 0) - 1

                        old_cluster = border_between_clusters

                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[last_elem, :] = last_cluster_vec
                            self.p_t_given_y[-(last_elem+1), :] = partner_last_cluster_vec

                            cur_card_T += 2

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1/(self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[last_elem, border_between_clusters] = 1
                                self.p_t_given_y[cardinality_Y-last_elem-1, cur_card_T-2-border_between_clusters-1] = 1

                            else:
                                self.p_t_given_y[last_elem, border_between_clusters+1] = 1
                                self.p_t_given_y[- (last_elem+1), cur_card_T - 2 - border_between_clusters-2] = 1
                                done_left_to_right = False

                            self.p_t_given_y[-(last_elem+1), -1] = 0
                            self.p_t_given_y[last_elem, - 2] = 0

                            cur_card_T -= 2

                    # check other direction
                    while not done_right_to_left:
                        done_right_to_left = True

                        # find first element in the cluster
                        first_elem = np.argmax(self.p_t_given_y[:, border_between_clusters + 1] > 0)

                        old_cluster = border_between_clusters + 1
                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[first_elem, :] = last_cluster_vec
                            self.p_t_given_y[cardinality_Y - first_elem - 1, :] = partner_last_cluster_vec

                            cur_card_T += 2

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[first_elem, border_between_clusters] = 1
                                self.p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 2 - border_between_clusters - 1] = 1
                                done_right_to_left = False
                            else:
                                self.p_t_given_y[first_elem, border_between_clusters + 1] = 1
                                self.p_t_given_y[cardinality_Y - first_elem-1, cur_card_T - 2 - border_between_clusters-2] = 1

                            self.p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 1] = 0
                            self.p_t_given_y[first_elem, cur_card_T - 2] = 0
                            cur_card_T -= 2

                end_mat = self.p_t_given_y

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

            p_t_given_y_mats[:, :, run] = self.p_t_given_y
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t

            p_ty = self.p_t_given_y[:, :self.cardinality_T] * p_y[:, np.newaxis]
            p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]

            I_YT[run] = mutual_information(p_ty)
            I_TX[run] = mutual_information(p_xt)

            ib_fct[run] = I_YT[run] / (-self.beta) + I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]

# ### Agglomerative Information Bottleneck Algorithm [Slo02]

class aIB(GenericIB):
    """This class can be used to perform the Agglomerative Information Bottleneck algorithm.
      Args:
      input parameter
          p_x_y                     input joint pdf, where x is the number of columns and y the number of rows
      IB related parameters
          cardinality_T
          beta                      is set to Inf for the sIB
          eps                       not used in this algorithm set to []
          nror                      number of runs
      Return:
      mutual information
          MI_XT                     mutual information of output I(X;T)
          MI_XY                     mutual information of input I(X;Y)
      output PDF_s
          p_t_given_y
          p_x_given_t
          p_t
      Note: The values can be accessed using the appropriate return functions of the class
        self.return_results         return all values
        self.display_MI             return the MIs graphically as well as percentage of maintained mutual information
      """

    def __init__(self, p_x_y_, card_T_):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], 5)
        self.name = 'agglomerative IB'

        self.cardinality_X=p_x_y_.shape[1]
        self.cardinality_Y=p_x_y_.shape[0]
        self.cost_mat= np.zeros((self.cardinality_Y,self.cardinality_Y))+np.inf

        self.merge_costs = dict()

        self.index_list = np.arange(self.cardinality_Y).tolist()
        self.index_vec = np.arange(self.cardinality_Y)

    def calc_merger_cost_pair(self,ind1,ind2):
        """Return the merger cost for putting one event in a cluster.
        Args:
            p_t: is a 1 x card_T array
            p_x_given_t: is a card_X x card_T array
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element

        ind1 = self.index_list.index(ind1)
        ind2 = self.index_list.index(ind2)

        p_t_bar = self.p_t[ind1] + self.p_t[ind2]

        pi1 = self.p_t[ind1] / p_t_bar
        pi2 = self.p_t[ind2] / p_t_bar

        self.cost_mat[ind1,ind2] = p_t_bar * (js_divergence(self.p_x_given_t[ind1, :], self.p_x_given_t[ind2, :], pi1, pi2) -
                              (pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)


        if self.cost_mat[ind1,ind2]<0:
            self.cost_mat[ind1, ind2] = 0

        return self.cost_mat[ind1,ind2]

    def calc_merger_cost_pair_vec(self,ind1,ind2):
        """Return the merger cost for putting one event in a cluster.
        Args:
            p_t: is a 1 x card_T array
            p_x_given_t: is a card_X x card_T array
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element

        p_t_bar = self.p_t[ind1] + self.p_t[ind2]

        pi1 = self.p_t[ind1] / p_t_bar
        pi2 = self.p_t[ind2] / p_t_bar

        self.cost_mat[ind1,ind2] = p_t_bar * (js_divergence(self.p_x_given_t[ind1, :], self.p_x_given_t[ind2, :], pi1, pi2) -
                              (pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)

        self.cost_mat[self.cost_mat < 0] = 0

    def calc_all_merge_costs(self):
        """
        This function is called only once, during initialization of Partition
        Subsequent calls operate on a subset of the data
        """

        ind1 = np.kron(np.arange(self.cardinality_Y), np.ones(self.cardinality_Y))
        ind2 = np.tile(np.arange(self.cardinality_Y),self.cardinality_Y)
        valid_combinations = ind1<ind2
        ind1 = ind1[valid_combinations].astype(int)
        ind2 = ind2[valid_combinations].astype(int)

        self.calc_merger_cost_pair_vec(ind1, ind2)


    def find_merge_pair(self):
        """
        Search all cluster pairs for the best pair to merge.
        Use the following criteria:
        1) Find pair(s) for which merge cost is minimized
        2) If multiple candidates from (1), find pair with smallest inter-cluster distance
        """

        min_pair = min(self.merge_costs, key=lambda x: self.merge_costs[x])

        min_val = self.merge_costs[min_pair]

        assert min_val == self.calc_merger_cost_pair(*min_pair)
        ties = [k for k, v in self.merge_costs.items() if v == min_val]

        if len(ties) > 1:
            min_pair = ties[0]
        return min_pair

    def find_merge_pair_vec(self):
        """
        Search all cluster pairs for the best pair to merge.
        Use the following criteria:
        1) Find pair(s) for which merge cost is minimized
        2) If multiple candidates from (1), find pair with smallest inter-cluster distance
        """

        min_pair = np.unravel_index(np.argmin(self.cost_mat), self.cost_mat.shape)

        return min_pair

    def merge(self,i,j):

        target, remove = sorted([i, j])

        target = self.index_list.index(target)
        remove = self.index_list.index(remove)

        del self.index_list[remove]

        # delete column in p(t|y)
        self.p_t_given_y[:,target] = self.p_t_given_y[:,target]+self.p_t_given_y[:,remove]
        self.p_t_given_y = np.delete(self.p_t_given_y, remove, axis=1)

        # delete row in p(x|t)
        # update p(t)

        self.p_t[target] = self.p_t[target] + self.p_t[remove]
        self.p_t = np.delete(self.p_t, remove, axis=0)

        self.p_x_and_t[target,:] = self.p_x_and_t[target,:] + self.p_x_and_t[remove,:]
        self.p_x_and_t = np.delete(self.p_x_and_t, remove, axis=0)

        self.p_x_given_t[target,:] = 1 / self.p_t[target] * (self.p_x_and_t[target,:] )
        self.p_x_given_t = np.delete(self.p_x_given_t, remove, axis=0)

    def merge_vec(self,i,j):

        target, remove = sorted([i, j])

        del self.index_list[remove]

        # delete column in p(t|y)
        self.p_t_given_y[:,target] = self.p_t_given_y[:,target]+self.p_t_given_y[:,remove]
        self.p_t_given_y = np.delete(self.p_t_given_y, remove, axis=1)

        # delete row in p(x|t)
        # update p(t)
        self.p_t[target] = self.p_t[target] + self.p_t[remove]
        self.p_t = np.delete(self.p_t, remove, axis=0)

        self.p_x_and_t[target,:] = self.p_x_and_t[target,:] + self.p_x_and_t[remove,:]
        self.p_x_and_t = np.delete(self.p_x_and_t, remove, axis=0)

        self.p_x_given_t[target,:] = 1 / self.p_t[target] * (self.p_x_and_t[target,:] )
        self.p_x_given_t = np.delete(self.p_x_given_t, remove, axis=0)

    def merge_next(self):
        """
        Iterate the AIB algorithm.
        Find best pair to merge, perform merge, update clusters and merge costs for next iteration
        """
        # Decide which pair of clusters to merge next
        min_pair = self.find_merge_pair_vec()

        # Execute merge
        self.merge_vec(*min_pair)
        """After merge, recompute costs related to the merged clusters
        Two steps:
            1) Update pointers to point to the merged pair (the min of min_pair)
            2) Process this list with clusters.calc_merge_cost
        """

        target, remove = sorted(min_pair)
        # entries are basically the row and column of remove
        self.cost_mat = np.delete(self.cost_mat, remove,axis=0)
        self.cost_mat = np.delete(self.cost_mat, remove,axis=1)

        #entries to update
        # all entries in target row and column that are not inf
        dummy_vec=np.arange(self.cost_mat.shape[0])
        # check column entries, i.e. ind1 is fixed and ind2 is determined by relevant_entries
        relevant_entries = np.logical_not( np.isinf(self.cost_mat[target,:]))
        ind2 = dummy_vec[relevant_entries]
        ind1 = target * np.ones(ind2.shape[0])
        self.calc_merger_cost_pair_vec(ind1.astype(int),ind2.astype(int))

        # check row entries, i.e. ind2 is fixed and ind1 is determined by relevant_entries
        relevant_entries = np.logical_not( np.isinf(self.cost_mat[:,target]))
        ind1 = dummy_vec[relevant_entries]
        ind2 = target * np.ones(ind1.shape[0])
        self.calc_merger_cost_pair_vec(ind1.astype(int), ind2.astype(int))

    def run_IB_algo(self):
        self.aIB_algo()

    def aIB_algo(self):
        """ This function tries to minimize the information bottleneck functional using the Agglomerative IB (Agg-IB) algorithm.
        This algorithm only allows for deterministic cluster mapping, meaning beta is always set to infinity."""

        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        self.I_XY_list = []
        self.I_XT_list = []

        self.p_x = self.p_x_y.sum(0)
        self.p_y = self.p_x_y.sum(1)

        # in the first step T is a perfect copy of y.
        self.p_t = p_y.copy()
        self.p_t_shortened = self.p_t.copy()

        p_x_and_t =  self.p_x_y.copy()
        self.p_x_and_t = self.p_x_y.copy()

        self.p_x_given_t = p_x_and_t / self.p_t[:,np.newaxis]
        self.p_x_given_t_shortened = self.p_x_given_t.copy()

        self.p_t_given_y = np.eye(self.cardinality_Y)


        self.calc_all_merge_costs()

        while self.p_x_given_t.shape[0]>self.cardinality_T:

            self.merge_next()

            p_xt = self.p_x_given_t * self.p_t[:, np.newaxis]

            self.I_XY_list.append(mutual_information(self.p_x_y))
            self.I_XT_list.append(mutual_information(p_xt))

        self.MI_XY = mutual_information(self.p_x_y)
        self.MI_XT = mutual_information(p_xt)

