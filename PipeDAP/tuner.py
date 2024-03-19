from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import random
import time
import numpy as np
from comm_core import rank, size, Communicator

num_init_points = 0
num_trials = 10
target_time = None

class Tuner(object):
    """Tuning parameter x to minimize the iteration time with Bayesian Optimization."""
    def __init__(self, x=32, bound=(1.0, 100.0), max_num_steps=num_trials):      
        self._current_point = x
        self._bound = bound
        self._max_num_steps = max_num_steps

        self._opt_point = None
        self._opt_iter_time = None
        # random search
        self._init_points = [random.random() * (bound[1]-bound[0]) + bound[0] 
                for _ in range(num_init_points)]
        # grid search
        self._init_points = [i * (bound[1]-bound[0]) / (num_init_points-1) + bound[0] 
                for i in range(num_init_points)]
        
        # empty init points
        #self._init_points = []
      
        self._num_steps = 0
        self._interval = 5
        self._timestamps = []
        self._warmup_record = True

        self._utility = UtilityFunction(kind='ei', kappa=0.0, xi=0.1)
        self._opt = BayesianOptimization(f=None, pbounds={'x': bound})
        
        #self._bo_cost = 0
        self._bo_cost = []

    def _put(self, x, iter_time):
        """Put param and iteration time into BO optimizer."""
        self._opt.register(params={'x': x}, target=-iter_time)

    def _next_point(self):
        if len(self._init_points) > 0:
            next_point = self._init_points.pop(0)
        else:
            stime = time.time()
            next_point = self._opt.suggest(self._utility)['x']
            #self._bo_cost += (time.time() - stime)
            self._bo_cost.append(time.time() - stime)
        return next_point

    def _record(self):
        self._timestamps.append(time.time())
        if len(self._timestamps) < self._interval:
            return None
      
        if self._warmup_record: # discard warmup time
            self._warmup_record = False
            return None
      
        durations = [self._timestamps[i] - self._timestamps[i-1] 
                     for i in range(3, len(self._timestamps))]
        self._timestamps = []
        return np.mean(durations)
      
    def opt_point(self):
        return self._opt_point, self._opt_iter_time
  
    def step(self):
        """Return new point for fine-tuning when it is ready, else return None."""
        if self._num_steps > self._max_num_steps:
            return None
      
        if self._num_steps == self._max_num_steps:
            self._num_steps += 1
            if rank() == 0:
                print("BO Tuning optimal param: %.4f, optimal iteration time %.4f" 
                        % (self._opt_point, self._opt_iter_time))
                #print("BO Tuning cost: %.4f" % (self._bo_cost / self._max_num_steps))
                print("BO Tuning cost:", np.mean(self._bo_cost))

            if self._current_point != self._opt_point:
                return self._opt_point
            else:
                return None

        # record time
        iter_time = self._record()
        if iter_time is None:
            return None
        
        if rank() == 0:
            print("BO Tuning step [%d], param: %.4f, iteration time: %.4f" % 
                    (self._num_steps, self._current_point, iter_time))

        # store the best result
        if self._opt_point is None or iter_time < self._opt_iter_time:
            self._opt_point = self._current_point
            self._opt_iter_time = iter_time

        # experiment: num of trials to reach target performance
        if target_time is not None and iter_time < target_time and self._num_steps > 0:
            if rank() == 0:
                print("BO Tuning step [%d], reaching target performance!" % self._num_steps)
            exit()

        # put current point and get next point
        self._put(self._current_point, iter_time)      
        next_point = self._next_point()
        self._current_point = next_point
        self._num_steps += 1
        return next_point
