import unittest

from autosklearn.smbo import AutoMLSMBO
from autosklearn.metrics import accuracy
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, Configuration


class TestSMBO(unittest.TestCase):

    def test_choose_next(self):
        configspace = ConfigurationSpace()
        configspace.add_hyperparameter(UniformFloatHyperparameter('a', 0, 1))
        configspace.add_hyperparameter(UniformFloatHyperparameter('b', 0, 1))

        dataset_name = 'foo'
        func_eval_time_limit = 15
        total_walltime_limit = 15
        memory_limit = 3072

        auto = AutoMLSMBO(
            config_space=None,
            dataset_name=dataset_name,
            backend=None,
            func_eval_time_limit=func_eval_time_limit,
            total_walltime_limit=total_walltime_limit,
            memory_limit=memory_limit,
            watcher=None,
            metric=accuracy
        )
        auto.config_space = configspace
        scenario = Scenario({
            'cs': configspace,
            'cutoff_time': func_eval_time_limit,
            'wallclock_limit': total_walltime_limit,
            'memory_limit': memory_limit,
            'run_obj': 'quality',
        })
        smac = SMAC(scenario)

        self.assertRaisesRegex(
            ValueError,
            'Cannot use SMBO algorithm on empty runhistory',
            auto.choose_next,
            smac
        )

        config = Configuration(configspace, values={'a': 0.1, 'b': 0.2})
        # TODO make sure the incumbent is always set?
        smac.solver.incumbent = config
        runhistory = smac.solver.runhistory
        runhistory.add(config=config, cost=0.5, time=0.5,
                       status=StatusType.SUCCESS)

        auto.choose_next(smac)
