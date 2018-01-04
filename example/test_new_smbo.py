import glob

import time

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from smac.facade.smac_facade import SMAC
from smac.optimizer.smbo import SMBO
from smac.optimizer import pSMAC

import autosklearn.classification


class AutoSklearnSMBO(SMBO):
    def run(self):
        """Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        """
        self.start()

        # Main BO loop
        while True:
            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_dirs=self.scenario.input_psmac_dirs,
                           configuration_space=self.config_space,
                           logger=self.logger)

            start_time = time.time()
            X, Y = self.rh2EPM.transform(self.runhistory)

            self.logger.debug("Search for next configuration")
            # get all found configurations sorted according to acq
            challengers = self.choose_next(X, Y)

            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)

            self.logger.debug("Intensify")

            self.incumbent, inc_perf = self.intensifier.intensify(
                challengers=challengers,
                incumbent=self.incumbent,
                run_history=self.runhistory,
                aggregate_func=self.aggregate_func,
                time_bound=max(self.intensifier._min_time, time_left))

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir)

            logging.debug(
                "Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                    self.stats.get_remaing_time_budget(),
                    self.stats.get_remaining_ta_budget(),
                    self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent



def get_smac_object_callback(
        scenario_dict,
        seed,
        ta,
        backend,
        metalearning_configurations,
):
    scenario = Scenario(scenario_dict)
    default_config = scenario.cs.get_default_configuration()
    initial_configurations = [default_config] + metalearning_configurations
    smac = SMAC(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        initial_configurations=initial_configurations,
        smbo_class=AutoSklearnSMBO,
    )
    smac.solver.backend = backend
    return smac


def main():
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_eips_example_tmp',
        output_folder='/tmp/autosklearn_eips_example_out',
        get_smac_object_callback=get_smac_object_callback,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=2,
    )
    automl.fit(X_train, y_train, dataset_name='digits')

    # Print the final ensemble constructed by auto-sklearn via ROAR.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
