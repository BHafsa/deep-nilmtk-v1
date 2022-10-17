NILM22_experiment = {
    'power': {'mains': ['active'], 'appliance': ['active']},
    'sample_rate': 8,
    'appliances': [],
    'artificial_aggregate': False,
    'DROP_ALL_NANS': True,
    'methods': {

    },
    'train': {
        'datasets': {
            'ukdale': {
                'path': None,
                'buildings': {
                    1: {
                        'start_time': '2015-01-04',
                        'end_time': '2015-01-15'
                    }
                }
            }
        }
    },
    'test': {
        'datasets': {
            'ukdale': {
                'path': None,
                'buildings': {
                    1: {
                        'start_time': '2015-04-16',
                        'end_time': '2015-04-30'
                    },
                }
            },

        },
        'metrics': ['mae', 'nde', 'rmse', 'f1score'],
    }

}

ukdale_0 = {'power': {'mains': ['active'], 'appliance': ['active']},
            'sample_rate': 8,
            'appliances': ['coffee maker'],
            'artificial_aggregate': False,
            'DROP_ALL_NANS': True,
            'methods': {},
            'train': {'datasets': {'ukdale': {'path': None,
                                              'buildings': {
                                                  1: {'start_time': '2013-09-07', 'end_time': '2013-11-13'}}}}},
            'test': {'datasets': {'ukdale': {'path': None,
                                             'buildings': {1: {'start_time': '2015-04-16',
                                                               'end_time': '2015-05-15'}}}},
                     'metrics': ['mae', 'nde', 'rmse', 'f1score']}}

# ukdale_1 = {'power': {'mains': ['active'], 'appliance': ['active']},
#             'sample_rate': 8,
#             'appliances': ['toaster'],
#             'artificial_aggregate': False,
#             'DROP_ALL_NANS': True,
#             'methods': {},
#             'train': {'datasets': {'ukdale': {'path': None,
#                                               'buildings': {1: {'start_time': '2013-11-13',
#                                                                 'end_time': '2014-02-03'}}}}},
#             'test': {'datasets': {'ukdale': {'path': None,
#                                              'buildings': {1: {'start_time': '2015-04-16',
#                                                                'end_time': '2015-05-15'}}}},
#                      'metrics': ['mae', 'nde', 'rmse', 'f1score']}}

# The dryer can also be integrated in the ukdale but rather here the problem
# is as follows two dryers are connected to the same meter and thus we do not have the
# recording for a single dryer but rather two dryers together.
# In the case of IADL this is not a problem since we are just interested in detecting if the
# the dryer is used or not as part of the laundry activity.
ukdale_1 = {'power': {'mains': ['active'], 'appliance': ['active']},
            'sample_rate': 8,
            'appliances': ['breadmaker',
                           'washing machine',
                           'microwave',
                           'washer dryer'],
            'artificial_aggregate': False,
            'DROP_ALL_NANS': True,
            'methods': {},
            'train': {'datasets': {'ukdale': {'path': None,
                                              'buildings': {1: {'start_time': '2014-03-17',
                                                                'end_time': '2014-07-21'}}}}},
            'test': {'datasets': {'ukdale': {'path': None,
                                             'buildings': {1: {'start_time': '2015-04-16',
                                                               'end_time': '2015-05-15'}}}},
                     'metrics': ['mae', 'nde', 'rmse', 'f1score']}}

ukdale_2 = {'power': {'mains': ['active'], 'appliance': ['active']},
            'sample_rate': 8,
            'appliances': ['kettle'],
            'artificial_aggregate': False,
            'DROP_ALL_NANS': True,
            'methods': {},
            'train': {'datasets': {'ukdale': {'path': None,
                                              'buildings': {1: {'start_time': '2014-08-16',
                                                                'end_time': '2014-10-04'}}}}},
            'test': {'datasets': {'ukdale': {'path': None,
                                             'buildings': {1: {'start_time': '2015-04-16',
                                                               'end_time': '2015-05-15'}}}},
                     'metrics': ['mae', 'nde', 'rmse', 'f1score']}}

### the boiler do not have recording for teh active power in the ukdale
### but rather contain different measures for the input and the target

# ukdale_4 =  {'power': {'mains': ['apparent'], 'appliance': ['active']},
#   'sample_rate': 8,
#   'appliances': ['boiler'],
#   'artificial_aggregate': False,
#   'DROP_ALL_NANS': True,
#   'methods': {},
#   'train': {'datasets': {'ukdale': {'path': None,
#      'buildings': {1: {'start_time': '2015-01-09',
#        'end_time': '2015-03-15'}}}}},
#   'test': {'datasets': {'ukdale': {'path': None,
#      'buildings': {1: {'start_time': '2015-04-16',
#        'end_time': '2015-05-15'}}}},
#    'metrics': ['mae', 'nde', 'rmse', 'f1score']}}

ukdale_4 = {'power': {'mains': ['active'], 'appliance': ['active']},
            'sample_rate': 8,
            'appliances': ['computer',
                           'dish washer',
                           'television',
                           'audio amplifier',
                           'oven'],
            'artificial_aggregate': False,
            'DROP_ALL_NANS': True,
            'methods': {},
            'train': {'datasets': {'ukdale': {'path': None,
                                              'buildings': {1: {'start_time': '2016-04-26',
                                                                'end_time': '2016-07-30'}}}}},
            'test': {'datasets': {'ukdale': {'path': None,
                                             'buildings': {1: {'start_time': '2015-04-16',
                                                               'end_time': '2015-05-15'}}}},
                     'metrics': ['mae', 'nde', 'rmse', 'f1score']}}
