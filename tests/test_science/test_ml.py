#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import MLEngineer, DataEngineer


class TestMLEngineer(object):
    def test_pca(self):
        data_eng = DataEngineer()
        data_eng.download()
        data_eng.clean()
        data_eng.transform()
        subset_df, *_ = data_eng.subset(geo="Japan")
        subset_df = subset_df.drop([self.N, self.S, self.C, self.CI, self.F, self.R], axis=1)
        # PCA
        eng = MLEngineer()
        pca_dict = eng.pca(X=subset_df, n_components=0.95)
        assert isinstance(pca_dict, dict)
        assert {"loadings", "PC", "explained_var", "topfeet"}.issubset(pca_dict)
