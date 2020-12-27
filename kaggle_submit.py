import re
import os
import pandas as pd

SUBMISSION_FOLDER = "submissions"
os.makedirs(SUBMISSION_FOLDER, exist_ok=True)

def save_for_submission(y, h5_test, fname=None):
    submission = pd.Series(data=y, index=h5_test["index_absolute"][:], name="sleep_stage")
    submissions = os.listdir(SUBMISSION_FOLDER)
    if fname is None:
        if len(submissions) == 0:
            fpath = os.path.join(SUBMISSION_FOLDER, "submission_1.csv")
        else:
            last = sorted(submissions)[-1]
            last_num = re.search("(\d+)\.csv", last).groups()[0]
            fpath = os.path.join(SUBMISSION_FOLDER, f"submission_{int(last_num)+1}.csv")
    else:
        fpath = os.path.join(SUBMISSION_FOLDER, fname)
    submission.to_csv(fpath, index_label='index')
    print(f"New submission file at {fpath}")
    return fpath

def send_submission_to_kaggle(submission_file, msg=""):
    os.system(f'kaggle competitions submit -c dreem-2-sleep-classification-challenge-2020 -f {submission_file} -m "{msg}"')


def submit_to_kaggle(y, h5_test, fname=None, msg=""):
    fname = save_for_submission(y, h5_test, fname)
    send_submission_to_kaggle(fname, msg=msg)
    

