# -*- encoding: utf-8 -*-
# Score library for NUMPY arrays
# ChaLearn AutoML challenge

# For regression:
# solution and prediction are vectors of numerical values of the same dimension

# For classification:
# solution = array(p,n) of 0,1 truth values, samples in lines, classes in columns
# prediction = array(p,n) of numerical scores between 0 and 1 (analogous
# to probabilities)

# Isabelle Guyon and Arthur Pesah, ChaLearn, August-November 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

from .classification_metrics import *
from .util import *
from .regression_metrics import *
from .factories import MetricFactory, KnownMetricFactory
from .loss_factory import MetricFromLossFactory\


def get_metric(metric, task_type=None):
    ''' Get Metric object containing metric name and a scoring function.

    Parameters
    ----------
    metric : string (internally known metric or package name) or a callable (as
        a custom metric)
    task_type : integer, autosklearn task type constant (is only used if the
        'metric' argument is a string, representing the name of the required
        known metric)

    Returns
    -------
    Metric object
    '''
    factory = MetricFactory()
    return factory.create(metric, task_type)

def get_metric_from_loss(loss):
    ''' Get Metric based on the provided loss function. The Metric, in this
        case, is a wrapper that inverts loss function value. Instead of the loss
        function minimization, autosklearn will maximize a value '1 - loss'.

    Parameters
    ----------
    loss : string (package name) or a callable (custom loss)

    Returns
    -------
    Metric object
    '''
    factory = MetricFromLossFactory()
    return factory.create(loss)

def get_all_known_metrics(task_type):
    ''' Returns a list of all known metrics for the corresponding task type

    Parameters
    ----------
    task_type : integer, autosklearn task type constant

    Returns
    list of Metric objects
    -------

    '''
    factory = KnownMetricFactory(task_type)
    return factory.get_all()
