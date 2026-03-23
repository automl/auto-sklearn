"""Mock smac.facade.smac_ac_facade module"""
import logging
import numpy as np

from .smac_facade import SMAC

class SMAC4AC(SMAC):
    def __init__(self, scenario, tae_runner=None, intensifier=None,
                 acquisition_function=None, model=None, runhistory=None,
                 runhistory2epm=None, initial_design=None, random_configuration_chooser=None,
                 acq_optimizer=None, initial_incumbent=None, stats=None, rng=None):
        super().__init__(scenario=scenario,
                         tae_runner=tae_runner,
                         intensifier=intensifier,
                         acquisition_function=acquisition_function,
                         model=model,
                         runhistory=runhistory,
                         runhistory2epm=runhistory2epm,
                         initial_design=initial_design,
                         random_configuration_chooser=random_configuration_chooser,
                         acq_optimizer=acq_optimizer,
                         initial_incumbent=initial_incumbent,
                         stats=stats,
                         rng=rng)

class SMAC4HPO(SMAC4AC):
    pass

class SMAC4BO(SMAC4AC):
    pass
