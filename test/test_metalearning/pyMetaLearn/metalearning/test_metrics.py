"""
def test_learned(self):
    kND = KNearestDatasets(metric='learned')
    rf = kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                 {233: self.runs[233], 234: self.runs[234]})

    self.assertEqual(kND._learned(self.anneal, self.krvskp), 1.5)
    self.assertEqual(kND._learned(self.anneal, self.labor), 1.5)


def test_learned_sparse(self):
    kND = KNearestDatasets(metric='learned')
    rf = kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                 {233: self.runs[233][0:2], 234: self.runs[234][1:3]})

    self.assertEqual(kND._learned(self.anneal, self.krvskp), 1.5)
    self.assertEqual(kND._learned(self.anneal, self.labor), 1.5)


def test_feature_selection(self):
    kND = KNearestDatasets(metric='mfs_l1',
                           metric_kwargs={'max_features': 1.0,
                                            'mode': 'select'})
    self.krvskp.name = 'kr-vs-kp'
    selection = kND.fit(pd.DataFrame([self.krvskp, self.labor, self.anneal]),
            {'kr-vs-kp': self.runs['krvskp'],
             'labor': self.runs['labor'],
             'anneal': self.runs['anneal']})
    self.assertEqual(1, selection.loc['number_of_classes'])
    self.assertEqual(1, selection.loc['number_of_features'])
    self.assertEqual(0, selection.loc['number_of_instances'])

def test_feature_weighting(self):
    kND = KNearestDatasets(metric='mfs_l1',
                           metric_kwargs={'max_features': 1.0,
                                            'mode': 'weight'})
    self.krvskp.name = 'kr-vs-kp'
    selection = kND.fit(pd.DataFrame([self.krvskp, self.labor, self.anneal]),
            {'kr-vs-kp': self.runs['krvskp'],
             'labor': self.runs['labor'],
             'anneal': self.runs['anneal']})
    self.assertEqual(type(selection), pd.Series)
    self.assertEqual(len(selection), 3)
"""
