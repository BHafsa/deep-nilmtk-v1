ukdale_experiment = {
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
                            'end_time': '2015-03-30'
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
                            'end_time': '2015-05-15'
                        },
                    }
                },

            },
            'metrics': ['mae', 'nde', 'rmse', 'f1score'],
        }

    }