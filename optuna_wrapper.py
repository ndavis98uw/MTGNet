import optuna
import MTGNet_run
import joblib
TRIAL = 'dualcolor_color'
label_dir = 'dualcolor_new'
is_color = True
version = 2
iteration = 0


def objective(trial):
    global iteration
    iteration += 1
    return MTGNet_run.network_run(label_dir, is_color, trial.suggest_int('Batch Size', 32, 88),
                                  trial.suggest_float('Learning Rate', .02, .15),
                                  trial.suggest_float('Weight Decay', .0002, .0015),
                                  trial.suggest_int('Epochs', 10, 25), str(version) + '.' + str(iteration), __name__)


if __name__ == "__main__":
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(objective, n_trials=16)
    joblib.dump(study, 'optuna_study_' + TRIAL + '.pkl')

