import torch
import numpy as np
#from acomm import rank, size, Communicator
from comm_core import rank, size, Communicator
import time


NUM_NEARBY_LAYERS=3 # DenseNet-201 10GbE 64-GPU
#NUM_NEARBY_LAYERS=10 # ResNet-50 10GbE 64-GPU
#NUM_NEARBY_LAYERS=2 # Bert-base 10GbE 64-GPU

TENSOR_FUSION_BUFFER_SIZE=1024*1024 # The number of elements

class TensorGroup:
    def __init__(self, tensor_names, single_layer=False, tensors=None, num_nearby_layers=NUM_NEARBY_LAYERS):
        self._tensor_names = tensor_names
        self._single_layer = single_layer
        self._num_nearby_layers = num_nearby_layers
        self._groups, self._group_indices_by_name = self._generate_groups_with_nearby_layers()
        self._group_buffers = {}
        self.reset_merge()

    def reset_merge(self):
        if self._group_buffers is not None:
            for k in self._group_buffers:
                buf = self._group_buffers[k]
                del buf
        self._group_flags = [[0]*len(g) for g in self._groups]
        self._group_keys = [':'.join(g) for g in self._groups]
        self._group_storages = [[None] * len(g) for g in self._groups]
        self._group_buffers = {}
        self._group_wait_times = [[0]*len(g) for g in self._groups]  # record tensor's wait-in-buffer time
        self._group_in_times = [[0]*len(g) for g in self._groups]  # record tensor's come-in timestamps

    def _generate_groups_with_nearby_layers(self):
        groups = []
        group_indices_by_name = {}
        current_group = []
        group_idx = 0
        for i, t in enumerate(self._tensor_names):
            group_indices_by_name[t] = (group_idx, len(current_group))
            current_group.append(t)
            if not self._single_layer and i % self._num_nearby_layers == 0 and i > 0:
                groups.append(current_group)
                current_group = []
                group_idx += 1
        if len(current_group) > 0:
            groups.append(current_group)
        if rank() == 0:
            print('# of Groups: ', len(groups))
        return groups, group_indices_by_name

    def _generate_groups_with_threshold(self):
        groups = []
        group_indices_by_name = {}
        current_group = []
        group_idx = 0
        for i, t in enumerate(self._tensor_names):
            group_indices_by_name[t] = (group_idx, len(current_group))
            current_group.append(t)
            if not self._single_layer and i % self._num_nearby_layers == 0 and i > 0:
                groups.append(current_group)
                current_group = []
                group_idx += 1
        if len(current_group) > 0:
            groups.append(current_group)
        return groups, group_indices_by_name

    def is_merged(self):
        return len(self._tensor_names) != len(self._groups)

    def get_group_index_by_name(self, name):
        group_idx, sub_idx = self._group_indices_by_name[name]
        return group_idx, sub_idx

    def clear_group_flags(self):
        self._group_flags = [[0]*len(g) for g in self._groups]

    def check_group_full(self, name):
        group_idx, sub_idx = self.get_group_index_by_name(name) 
        if np.sum(self._group_flags[group_idx]) < len(self._group_flags[group_idx]):
            return False
        return True

    def push_tensor(self, name, tensor):
        group_idx, sub_idx = self.get_group_index_by_name(name) 
        group_key = self._group_keys[group_idx]
        numel = tensor.numel()
        self._group_flags[group_idx][sub_idx] = 1
        #if rank() == 0:
        #    print('group index by name key', self._group_indices_by_name.keys())
        #    print('group flags', self._group_flags)
        self._group_storages[group_idx][sub_idx] = tensor
        self._group_in_times[group_idx][sub_idx] = time.time()

        if self.check_group_full(name):
            time_out = time.time()
            if group_key not in self._group_buffers:
                total_size = 0
                for t in self._group_storages[group_idx]:
                    total_size += t.numel()
                self._group_buffers[group_key] = tensor.new_zeros(total_size)
            buf = self._group_buffers[group_key]
            offset = 0
            for t in self._group_storages[group_idx]:
                numel = t.numel()
                buf.data[offset:offset+numel].copy_(t.view(numel))
                offset += numel
            
            #time_out = time.time()
            for i, time_in in enumerate(self._group_in_times[group_idx]):
                self._group_wait_times[group_idx][i] += (time_out - time_in) * 1000 # ms
            return group_key, buf

        return name, None

    def pull_alltensors(self):
        for group_key in self._group_buffers:
            names = group_key.split(':')
            group_idx, sub_idx = self.get_group_index_by_name(names[0]) 
            buf = self._group_buffers[group_key]

            offset = 0
            for t in self._group_storages[group_idx]:
                numel = t.numel()
                t.copy_(buf.data[offset:offset+numel].view(t.shape))
                offset += numel 

    def update_groups(self, sizes, times, symmetric=False, reverse=False):
        if self._single_layer:
            return
        self._groups, self._group_indices_by_name, max_saved = self._generate_groups_spd(self._tensor_names, sizes, times, symmetric, reverse)
        self.reset_merge()
        torch.cuda.empty_cache()

    def update_reduce_groups(self, tensor_group_names, sizes, times, symmetric=False, reverse=False):
        if self._single_layer:
            return
        idx = 0
        groups = []
        group_indices_by_name = {}
        min_iter = 0
        for group in tensor_group_names:
            current_group_sizes = sizes[idx:idx+len(group)]
            current_times = times[idx:idx+len(group)]
            current_gen_groups, current_gen_group_indices_by_name, mi = self._generate_groups_spd(group, current_group_sizes, current_times, symmetric, reverse, idx)
            min_iter += mi
            groups = groups+current_gen_groups
            group_indices_by_name.update(current_gen_group_indices_by_name)
            idx += len(current_gen_groups)
        self._groups, self._group_indices_by_name = groups, group_indices_by_name
        if rank() == 0:
            logger.info('Total min iter: %f', min_iter)
        self.reset_merge()
        torch.cuda.empty_cache()

    def update_groups_with_configured_groups(self, tensor_group_names):
        if self._single_layer:
            return
        groups = []
        group_indices_by_name = {}
        group_idx = 0
        for i, tensor_list in enumerate(tensor_group_names):
            current_group = []
            for t in tensor_list:
                group_indices_by_name[t] = (group_idx, len(current_group))
                current_group.append(t)
            groups.append(current_group)
            group_idx += 1
        self._groups = groups
        self._group_indices_by_name = group_indices_by_name
        self.reset_merge()
        torch.cuda.empty_cache()

    def update_groups_with_flags(self, flags):
        groups = []
        group_indices_by_name = {}
        current_group = []
        group_idx = 0
        for i, t in enumerate(self._tensor_names):
            group_indices_by_name[t] = (group_idx, len(current_group))
            current_group.append(t)
            if flags[i] > 0: # close the group
                groups.append(current_group)
                current_group = []
                group_idx += 1
        if len(current_group) > 0:
            groups.append(current_group)
        self._groups = groups
        self._group_indices_by_name = group_indices_by_name
        self.reset_merge()
        torch.cuda.empty_cache()

    def get_wait_time(self):
        return self._group_wait_times

    def reset_wait_time(self):
        # update groups before reset wait times
        self._group_wait_times = [[0]*len(g) for g in self._groups]
        self._group_in_times = [[0]*len(g) for g in self._groups]



class CollectiveOp:
    REDUCE='REDUCE'
    BCAST ='BCAST'
    ALLREDUCE ='ALLREDUCE'
    REDUCE_SCATTER ='REDUCE_SCATTER'
    ALL_GATHER ='ALL_GATHER'

class MergedCommCollective:
    def __init__(self, tensor_names=None, prefix='flag', merge=False, single_layer=False, symmetric=False, fp16=False, op=CollectiveOp.REDUCE, nstreams=1):
        self._tensor_names = tensor_names
        self.merge = merge
        self.single_layer = single_layer
        self.symmetric = symmetric
        self.prefix = prefix
        self.fp16 = fp16
        self.tensor_group_names = None
        self.op = op
        self._current_stream = torch.cuda.current_stream()
        if tensor_names is not None:
            self.init_tensor_group(tensor_names)
        self.nstreams = nstreams
        self.merged_comm = Communicator(nstreams)

        self._name_tensors = {}
        self.handles = []

    def init_tensor_group(self, tensor_names, num_nearby_layers=NUM_NEARBY_LAYERS):
        self.tensor_names = tensor_names
        if self.merge:
            self._tensor_group = TensorGroup(tensor_names, single_layer=self.single_layer, num_nearby_layers=num_nearby_layers) 
        else:
            self._tensor_group = None

    def update_tensor_fusion(self, tensor_group_names):
        """
        tensor_group_names: [['tensor1', 'tensor2'], ['tensor3', 'tensor4'], ...]
        """
        if self._tensor_group is None:
            return
        self._tensor_group.update_groups_with_configured_groups(tensor_group_names)
        self.tensor_group_names = tensor_group_names

    def update_groups(self, tensor_group_names, sizes, times, reverse=False):
        if self.merge and self._tensor_group:
            self._tensor_group.update_reduce_groups(tensor_group_names, sizes, times, self.symmetric, reverse=reverse)
            self.merge = self._tensor_group.is_merged()

    def update_tensor_fusion_wf(self, threshold):
        """update tensor fusion with tensor waiting time"""
        if self._tensor_group is None:
            return
        group_wait_times = self._tensor_group.get_wait_time()

        # case (1): split one single layer
        if len(group_wait_times) == 1:
            split_threshold = threshold
            split_flags = []
            for wt in reversed(group_wait_times[0]):
                if wt > split_threshold:
                    split_flags.append(1.)
                    split_threshold += threshold
                    #split_threshold += wt
                else:
                    split_flags.append(0.)
            split_flags = [t for t in reversed(split_flags)]
            split_flags_tensor = torch.tensor(split_flags).cuda()
            #if rank() == 0:
            #    print(split_flags_tensor)
            handle = self.merged_comm.bcast(split_flags_tensor, 0)
            self.merged_comm.syncStream(handle)
            if rank() == 0:
                print("Tensor fusion with flags:", split_flags_tensor)
            self._tensor_group.update_groups_with_flags(split_flags_tensor)

        self._tensor_group.reset_wait_time()

    def get_wait_time(self):
        return self._tensor_group.get_wait_time()

    def reset_wait_time(self):
        return self._tensor_group.reset_wait_time()

    def sync_handle(self, handle):
        self.merged_comm.syncStream(handle)

    def collective_async_(self, name, tensor, r):
        handle = None
        if self.merge:
            assert self._tensor_group is not None, 'self._tensor_group has not been initialized'
            if self.symmetric:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
            else:
                comm_tensor = tensor

            self._name_tensors[name] = (tensor, comm_tensor, r)
            new_name, new_tensor = self._tensor_group.push_tensor(name, comm_tensor)
            if new_tensor is not None:
                #current_stream = torch.cuda.current_stream()
                #current_stream.synchronize()
                self._current_stream.wait_stream(self._current_stream)

                if self.op == CollectiveOp.REDUCE:
                    handle = self.merged_comm.reduce(new_tensor, r)
                elif self.op == CollectiveOp.BCAST:
                    handle = self.merged_comm.bcast(new_tensor, r)
                self.handles.append((handle, comm_tensor, tensor, r))
        else:
            if self.symmetric:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
            else:
                comm_tensor = tensor
            self._name_tensors[name] = (tensor, comm_tensor, r)

            #current_stream = torch.cuda.current_stream()
            #current_stream.synchronize()
            self._current_stream.wait_stream(self._current_stream)
            if self.op == CollectiveOp.REDUCE:
                handle = self.merged_comm.reduce(tensor, r)
            elif self.op == CollectiveOp.BCAST:
                handle = self.merged_comm.bcast(tensor, r)
            elif self.op == CollectiveOp.ALLREDUCE:
                handle = self.merged_comm.allReduce(tensor)
            self.handles.append((handle, comm_tensor, tensor, r))
        return handle

    def synchronize(self):
        self.merged_comm.synchronize()
        if self.merge:
            self._tensor_group.pull_alltensors()
            self._tensor_group.clear_group_flags()
        for name in self._name_tensors:
            tensor, comm_tensor, r = self._name_tensors[name]
            if r != rank():
                continue
            if self.symmetric:
                lower_indices = torch.tril_indices(tensor.shape[0], tensor.shape[1], device=tensor.device)
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[1], device=tensor.device)
                tensor[upper_indices[0], upper_indices[1]] = comm_tensor
                tensor[lower_indices[0], lower_indices[1]] = tensor.t()[lower_indices[0], lower_indices[1]]
            else:
                pass
            tensor.div_(size())
        self._name_tensors.clear()
        self.handles.clear()


class MergedCommReduce:
    def __init__(self, tensor_names=None, prefix='flag', merge=False, single_layer=False, symmetric=False, fp16=False, op=CollectiveOp.REDUCE):
        self._tensor_names = tensor_names
        self.merge = merge
        self.single_layer = single_layer
        self.symmetric = symmetric
        self.prefix = prefix
        self.fp16 = fp16
        self.tensor_group_names = None
        self.initialized = False
        if tensor_names is not None:
            self.init_tensor_group(tensor_names)
        nstreams = 1
        self.merged_comm = Communicator(nstreams)
        self._current_stream = torch.cuda.current_stream()
        self.op = op

        self._name_tensors = {}
        self.handles = []

    def init_tensor_group(self, tensor_names, num_nearby_layers=NUM_NEARBY_LAYERS):
        self.tensor_names = tensor_names
        if self.merge:
            self._tensor_group = TensorGroup(tensor_names, single_layer=False, num_nearby_layers=num_nearby_layers) 
        else:
            self._tensor_group = None
        self.initialized = True

    def update_tensor_fusion(self, tensor_group_names):
        """
        tensor_group_names: [['tensor1', 'tensor2'], ['tensor3', 'tensor4'], ...]
        """
        if self._tensor_group is None:
            return
        self._tensor_group.update_groups_with_configured_groups(tensor_group_names)
        self.tensor_group_names = tensor_group_names

    def update_groups(self, tensor_group_names, sizes, times, reverse=False):
        if self.merge and self._tensor_group:
            self._tensor_group.update_reduce_groups(tensor_group_names, sizes, times, self.symmetric, reverse=reverse)
            self.merge = self._tensor_group.is_merged()

    def reduce_async_(self, name, tensor, root_rank):
        if self.initialized and self.merge:
            assert self._tensor_group is not None, 'self._tensor_group has not been initialized'
            if self.symmetric:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
            else:
                comm_tensor = tensor

            self._name_tensors[name] = (tensor, comm_tensor, root_rank)
            new_name, new_tensor = self._tensor_group.push_tensor(name, comm_tensor)
            if new_tensor is not None:
                #current_stream = torch.cuda.current_stream()
                #current_stream.synchronize()
                self._current_stream.wait_stream(self._current_stream)
                if self.op == CollectiveOp.REDUCE:
                    handle = self.merged_comm.reduce(new_tensor, root_rank)
                elif self.op == CollectiveOp.BCAST:
                    handle = self.merged_comm.bcast(new_tensor, root_rank)
                elif self.op == CollectiveOp.ALLREDUCE:
                    handle = self.merged_comm.allReduce(new_tensor)
                self.handles.append((handle, comm_tensor, tensor, root_rank))
        else:
            if self.symmetric:
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[0], device=tensor.device)
                comm_tensor = tensor[upper_indices[0], upper_indices[1]]
            else:
                comm_tensor = tensor
            self._name_tensors[name] = (tensor, comm_tensor, root_rank)
            #current_stream = torch.cuda.current_stream()
            #current_stream.synchronize()
            self._current_stream.wait_stream(self._current_stream)

            if self.op == CollectiveOp.REDUCE:
                handle = self.merged_comm.reduce(comm_tensor, root_rank)
            elif self.op == CollectiveOp.BCAST:
                handle = self.merged_comm.bcast(comm_tensor, root_rank)
            elif self.op == CollectiveOp.ALLREDUCE:
                handle = self.merged_comm.allReduce(comm_tensor)
            self.handles.append((handle, comm_tensor, tensor, root_rank))

    def synchronize(self):
        self.merged_comm.synchronize()
        if self.initialized and self.merge:
            self._tensor_group.pull_alltensors()
            self._tensor_group.clear_group_flags()
        for name in self._name_tensors:
            tensor, comm_tensor, r= self._name_tensors[name]
            if r != rank():
                continue
            if self.symmetric:
                lower_indices = torch.tril_indices(tensor.shape[0], tensor.shape[1], device=tensor.device)
                upper_indices = torch.triu_indices(tensor.shape[0], tensor.shape[1], device=tensor.device)
                tensor[upper_indices[0], upper_indices[1]] = comm_tensor
                tensor[lower_indices[0], lower_indices[1]] = tensor.t()[lower_indices[0], lower_indices[1]]
            else:
                pass
            tensor.div_(size())
        self._name_tensors.clear()
        self.handles.clear()

class CommReduceScatter:
    def __init__(self, tensor_names=None, op=CollectiveOp.REDUCE_SCATTER):
        self._tensor_names = tensor_names
        nstreams = 1
        self.merged_comm = Communicator(nstreams)
        self._current_stream = torch.cuda.current_stream()
        self.op = op

        self._name_tensors = {}
        self.handles = []
    
    def init_tensor_group(self, tensor_names, num_nearby_layers=NUM_NEARBY_LAYERS):
        pass

    def collective_async_(self, name, pad_tensor, shard_tensor):
        self._name_tensors[name] = (pad_tensor, shard_tensor)
        #current_stream = torch.cuda.current_stream()
        #current_stream.synchronize()
        self._current_stream.wait_stream(self._current_stream)

        if self.op == CollectiveOp.REDUCE_SCATTER:
            handle = self.merged_comm.reduceScatter(pad_tensor, shard_tensor)
        elif self.op == CollectiveOp.ALL_GATHER:
            handle = self.merged_comm.allGather(shard_tensor, pad_tensor)
        else:
            raise TypeError
        self.handles.append((handle, shard_tensor, pad_tensor))
        return handle

    def synchronize(self):
        self.merged_comm.synchronize()
        self._name_tensors.clear()
        self.handles.clear()
