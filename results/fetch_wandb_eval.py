import wandb
import argparse
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--eval_batch_size", type=int, help="Batch size used to evaluate.")
    parser.add_argument("--classifier_model", type=str, help="Substring of the classifier model name.")
    parser.add_argument("--eval_classifier_model", type=str, help="Substring of the evaluation classifier model name.")
    parser.add_argument("--trainer_config", type=str, help="Substring of the trainer name.")
    parser.add_argument("--eval_trainer_config", type=str, help="Substring of the evaluation trainer name.")
    args = parser.parse_args()

    wandb_project = "Point Evaluation, %s" % args.dataset
    api = wandb.Api(timeout=100)
    runs = api.runs("%s/%s" % (args.wandb_name, wandb_project))
    for run in runs:
        config = run.config
        if (args.classifier_model == run.config["classifier_model_config"]) and \
                (args.eval_classifier_model in run.config["eval_classifier_model_config"]) and \
                (args.trainer_config == run.config["trainer_config"]) and \
                (args.eval_trainer_config in run.config["eval_trainer_config"]) and \
                (args.eval_batch_size == run.config["eval_batch_size"]):
            run_dict = {}
            print(run.name)
            for row in run.scan_history():
                if "Num Labeled" in row and row["Num Labeled"] is not None:
                    for key in row:
                        if key in run_dict:
                            run_dict[key].append(row[key])
                        else:
                            run_dict[key] = [row[key]]

            path_name = "./%s_eval/bs_%d/proxy_%s/model_%s/trainer_%s_eval_trainer_%s" % (
                args.dataset, args.eval_batch_size, args.classifier_model, args.eval_classifier_model,
                args.trainer_config, args.eval_trainer_config)
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            with open("%s/%s_%d.pkl" % (path_name, run.config["strategy_config"], run.config["seed"]), "wb") as file:
                pickle.dump(run_dict, file)
