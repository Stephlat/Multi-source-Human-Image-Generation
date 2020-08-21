import pandas as pd
from cmd import args
import pose_transform
import pose_utils
from itertools import permutations
import random

args = args()

def filter_not_valid(df_keypoints):
    def check_valid(x):
        kp_array = pose_utils.load_pose_cords_from_strings(x['keypoints_y'], x['keypoints_x'])
        distractor = x['name'].startswith('-1') or x['name'].startswith('0000')
        return pose_transform.check_valid(kp_array) and not distractor
    return df_keypoints[df_keypoints.apply(check_valid, axis=1)].copy()



def make_n_tuples(df,args):
    persons = df.apply(lambda x: '_'.join(x['name'].split('_')[0:1]), axis=1)
    df['person'] = persons
    n=args.nb_inputs
    names=['from_'+str(i) for i in range(n)]+["to"]
    listFrTo={name:[] for name in names}

    for person in pd.unique(persons):
        permu=list(permutations(df[df['person'] == person]['name'], n+1))
        nbim=len(df[df['person'] == person]['name'])
        nbpairs=nbim*(nbim-1)
        if len(permu)>2:
            permu=[ permu[i] for i in random.sample(xrange(len(permu)), nbpairs) ]
            n_tuples = zip(*permu)
            for i,name in enumerate(names):
                listFrTo[name]+= list(n_tuples[i])

    tuple_df = pd.DataFrame(index=range(len(listFrTo["to"])))
    for name in listFrTo.keys():
        tuple_df[name] = listFrTo[name]
    return tuple_df


if __name__ == "__main__":
    df_keypoints = pd.read_csv(args.annotations_file_train, sep=':')
    df = filter_not_valid(df_keypoints)

    assert(args.nb_inputs>=4)
    
    print ('Compute tuple dataset for train...')
    n_tuples_df_train = make_n_tuples(df,args)
    print ('Number of n_tuples: %s' % len(n_tuples_df_train))
    n_tuples_df_train.to_csv(args.file_train, index=False)

    print ('Compute tuple dataset for test...')
    df_keypoints = pd.read_csv(args.annotations_file_test, sep=':')
    df = filter_not_valid(df_keypoints)
    n_tuples_df_test = make_n_tuples(df,args)
    n_tuples_df_test = n_tuples_df_test.sample(n=min(args.images_for_test, n_tuples_df_test.shape[0]), replace=False, random_state=0)
    print ('Number of n_tuples: %s' % len(n_tuples_df_test))
    n_tuples_df_test.to_csv(args.file_test, index=False)
