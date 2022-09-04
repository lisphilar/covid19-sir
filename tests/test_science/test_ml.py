#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import MLEngineer, DataEngineer, Term


class TestMLEngineer(object):
    def _subset(self):
        data_eng = DataEngineer()
        data_eng.download()
        data_eng.clean()
        data_eng.transform()
        return data_eng.subset(geo="Japan")[0]

    def test_pca(self):
        subset_df = self._subset()
        subset_df = subset_df.drop([Term.N, Term.S, Term.C, Term.CI, Term.F, Term.R], axis=1)
        # PCA
        eng = MLEngineer()
        pca_dict = eng.pca(X=subset_df, n_components=0.95)
        assert isinstance(pca_dict, dict)
        assert {"loadings", "PC", "explained_var", "topfeat"}.issubset(pca_dict)

    def test_forecast(self):
        subset_df = self._subset()
        Y = subset_df.loc[:, [Term.C, Term.F, Term.R]]
        subset_df = subset_df.drop([Term.N, Term.S, Term.C, Term.CI, Term.F, Term.R], axis=1)
        # PCA
        eng = MLEngineer()
        pca_dict = eng.pca(X=subset_df, n_components=0.95)
        X = pca_dict["PC"].copy()
        # Forecasting without other information
        X_pred = eng.forecast(Y=X, days=30, X=None)
        # Forecasting with other information
        Y_pred = eng.forecast(Y=Y, days=30, X=X_pred)
        assert X_pred.index[-1] == Y_pred.index[-1]
