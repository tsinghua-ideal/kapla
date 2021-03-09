import numpy as np
import sys
import time
try:
    import xgboost as xgb
except ImportError:
    xgb = None

class XGBoostCostModel(object):
    """XGBoost as cost model"""
    def __init__(self, n_threads=1, log_interval=100):
        if xgb is None:
            raise RuntimeError("XGBoost is required for XGBoostCostModel. "
                               "Please install its python package first. "
                               "Help: (https://xgboost.readthedocs.io/en/latest/) ")

        self.xgb_params = {
            'max_depth': 3,
            'gamma': 0.0001,
            'min_child_weight': 1,
            'subsample': 1.0,
            'eta': 0.3,
            'lambda': 1.00,
            'alpha': 0,
            'objective': 'rank:pairwise',
            'nthread': n_threads,
        }

        # self.xgb_params['silent'] = 1
        self.xgb_params['verbosity'] = 0
        self.log_interval = log_interval
        self.n_threads = n_threads

    def fit(self, xs, ys, plan_size):
        tic = time.time()
        # print("xs: ", xs)
        # print("ys: ", ys)
        x_train = np.array(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        # print("x_train: ")
        # print(x_train)
        # print("y_train: ")
        # print(y_train)
        index = np.random.permutation(len(x_train))
        # print("index:")
        # print(index)
        dtrain = xgb.DMatrix(x_train[index], y_train[index])

        # sys.stderr.write("x_train_len: {}".format(len(x_train)))
        # self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=8000,
        #                      callbacks=[custom_callback(
        #                                 stopping_rounds=20,
        #                                 metric='tr-a-recall@' + str(plan_size),
        #                                 evals=[(dtrain, 'tr')],
        #                                 maximize=True,
        #                                 fevals=[
        #                                     xgb_average_recalln_curve_score(
        #                                     plan_size),
        #                                 ],
        #                                 verbose_eval=self.log_interval)])
        self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=10,
                             feval=xgb_average_recalln_curve_score(plan_size))
        # sys.stderr.write("train elapsed: {} obs: {} error: {}\n".format(
                    #  time.time() - tic, len(xs),
                    #  len(xs) - np.sum(valid_index)))

        return True

    def predict(self, feas, output_margin=False):
        # print("predict:")
        # print(feas)
        # print(len(feas))
        # print(len(feas[0]))
        if not isinstance(feas, np.ndarray):
            feas = np.array(feas)
        # print(type(feas))
        dtest = xgb.DMatrix(feas)

        return self.bst.predict(dtest, output_margin=output_margin)


def xgb_average_recalln_curve_score(N):
    """Evaluate average recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@" + str(N), np.sum(curve[:N]) / N
    return feval

def custom_callback(stopping_rounds, metric, fevals, evals=(), log_file=None,
                    maximize=False, verbose_eval=True):
    """
    Callback function for xgboost to support multiple custom evaluation functions.
    """
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric
    from xgboost.training import aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        """internal function"""
        bst = env.model

        state['maximize_score'] = maximize
        state['best_iteration'] = 0
        if maximize:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(':') for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        infos = ["XGB iter: %3d" % i]
        for item in eval_res:
            if 'null' in item[0]:
                continue
            infos.append("{}: {}".format(item[0], item[1]))

        if not isinstance(verbose_eval, bool) and verbose_eval and i % verbose_eval == 0:
            sys.stderr.write("\t\n".join(infos))
        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + '\n')

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d] %s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in eval_res]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose_eval and env.rank == 0:
                sys.stderr.write("XGB stopped. Best iteration: " + str(best_msg) + "\n")
            raise EarlyStopException(best_iteration)

    return callback

def get_rank(values):
    """get rank of items
    Parameters
    ----------
    values: Array
    Returns
    -------
    ranks: Array of int
        the rank of this item in the input (the largest value ranks first)
    """
    tmp = np.argsort(-values)
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(tmp))
    return ranks

def recall_curve(trial_ranks, top=None):
    """
    if top is None, f(n) = sum([I(rank[i] < n) for i < n]) / n
    if top is K,    f(n) = sum([I(rank[i] < K) for i < n]) / K
    Parameters
    ----------
    trial_ranks: Array of int
        the rank of i th trial in labels
    top: int or None
        top-n recall
    Returns
    -------
    curve: Array of float
        function values
    """
    if not isinstance(trial_ranks, np.ndarray):
        trial_ranks = np.array(trial_ranks)

    ret = np.zeros(len(trial_ranks))
    if top is None:
        for i in range(len(trial_ranks)):
            ret[i] = np.sum(trial_ranks[:i] <= i) / (i+1)
    else:
        for i in range(len(trial_ranks)):
            ret[i] = 1.0 * np.sum(trial_ranks[:i] < top) / top
    return ret
