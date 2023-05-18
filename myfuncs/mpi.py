import numpy as np
import sys
from mpi4py import MPI 

def autoMPI(total_tasks, level='basic'):
    """Automate MPI distribution.

    Parameters
    ----------
    total_tasks : list
        All of the tasks to split among the processes
    level : str, optional
        Describes how much MPI information you want returned, by default 'basic'. Options are 'basic', 'lazy', 'advanced'

    Returns
    -------
    list
        if level == 'basic':
            returns [comm, current_rank, subset_of_tasks_for_that_rank]
        elif level == 'lazy':
            returns everything that 'basic' returns, plus [Nranks, tasks_per_rank]
        elif level == 'advanced':
            returns everything that 'lazy' returns, plus [displacement_indices]

    Raises
    ------
    ValueError
        Valid 'level' options are ['basic', 'lazy', 'advanced']
    """
    #Level Options
    level_options['basic'] = 3
    level_options['lazy'] = level_options['basic'] + 2
    level_options['lazy'] = level_options['lazy'] + 1
    if level not in level_options.keys():
        raise ValueError("'level' options are ['basic', 'lazy', 'advanced']")

    #Basics
    comm = MPI.COMM_WORLD
    current_rank = comm.Get_rank()
    Nranks = comm.Get_size()

    #Break Up Work
    tasks_per_rank, displacements = calcMPI(len(total_tasks), Nranks)
    subset = total_tasks[displacements[current_rank] : displacements[current_rank+1]]

    return_list = [comm, current_rank, subset, Nranks, tasks_per_rank, displacements]

    return return_list[:level_options[level]]


def calcMPI(Nsims, Nprocs):
    """
    Calculates number of sims for each rank and the indices for breaking up the list of sim names.

    Args:
        Nsims (int): Number of things to break up over the multiple processes
        Nprocs (int): Number of ranks
    """

    #Initialize
    try:
        rows_per_rank = np.zeros((len(Nsims), Nprocs), dtype= int)
        displacement = np.zeros((len(Nsims), Nprocs+1), dtype= int)
    except TypeError:
        rows_per_rank = np.zeros(Nprocs, dtype= int)
        displacement = np.zeros(Nprocs+1, dtype= int)

    #Number of Sims Per Rank
    base_work, remainder = divmod(Nsims, Nprocs)    
    rows_per_rank[:] = base_work
    if remainder > 0: rows_per_rank[-remainder:] += 1         # rank 0 never gets extra work

    #Add ending Indices For Each Rank
    displacement[1:] = np.cumsum(rows_per_rank)
    displacement[-1] += 1      # so that the final sim is included
    
    return rows_per_rank, displacement


def mpiReduce(comm, current_rank, array, root_rank=0):
    """Just a wrapper for MPI Reduce, but ignores NaN's.

    Parameters
    ----------
    comm : MPI_Comm
        MPI communicator
    current_rank : int
        The current rank
    array : float
        Array to broadcast. Must be a numpy array
    root_rank : int, optional
        Main rank that does the assembling, by default 0

    Returns
    -------
    array or None
        The reduced numpy array if called on the main rank; None otherwise
    """
    #Initialize Receiving Buffer 
    recbuf = None
    if current_rank == root_rank:
        Nranks = comm.Get_size()
        tot_shape = [Nranks] + list(array.shape)
        recbuf = np.zeros(tot_shape)

    comm.Gather(array, recbuf, root= root_rank)

    if current_rank == root_rank:
        return np.nansum(recbuf, axis=0)
    else:
        return None


def printSimTime(start, end, rank, simnum, rank_Nsims, tot_time):
    """Prints time it took to complete a job on each rank and estimate remaining total time assuming that all ranks take the same amount of time (the idea is that you have multiple ranks performing a series of tasks).

    Parameters
    ----------
    start : float
        start timestamp in seconds (e.g. time.time())
    end : float
        end timestamp in seconds (e.g. time.time())
    rank : int
        Current process ID
    simnum : int
        Current counter value for the current task of the current rank
    rank_Nsims : int
        Total number of tasks on current rank
    tot_time : float
        Running total time

    Returns
    -------
    float
        Updated total time (doesn't take into account the time it took to call and run this function)
    """
    sim_time = end - start 
    tot_time += sim_time

    print(f'\nRank {rank} completed sim {simnum} / {rank_Nsims} in {sim_time:.0f} seconds.')
    sys.stdout.flush()

    if simnum != rank_Nsims:
        remaining_time = tot_time * rank_Nsims/simnum - tot_time
        remain_min, remain_sec = divmod(remaining_time, 60)
        remain_sec = round(remain_sec)

        print(f'Approximately {remain_min :.0f} minutes and {remain_sec} seconds remaining.')
        sys.stdout.flush()

    return tot_time


def printTotalTime(start, end, hourFlag=False, Nthings=0, type='sims'):
    """Calculates and prints out the total time it took to complete multiple (or 1) tasks. Prints time in min and sec (and hours, optionally).

    Parameters
    ----------
    start : float
        start timestamp in seconds (e.g. time.time())
    end : float
        end timestamp in seconds (e.g. time.time())
    hourFlag : bool, optional
        Do you want the time to be terms of hours too?. By default False
    Nthings : int, optional
        Number of tasks, by default 0
    type : str, optional
        Name of the type of task, by default 'sims'
    """

    time_min, time_sec = divmod(end-start, 60)
    
    if not hourFlag:
        #Cleanup
        time_sec = round(time_sec) 

        #Print
        if Nthings > 0:
            print(f'\nTook {time_min:.0f} min and {time_sec} sec for {Nthings} ' + type)
        else:
            print(f'\nTook {time_min:.0f} min and {time_sec} sec')

    elif hourFlag:
        #Calculations
        time_hour, time_min = divmod(time_min, 60)
        time_min = round(time_min) 

        #Print
        if Nthings > 0:
            print(f'\nTook {time_hour:.0f} hour and {time_sec} sec for {Nthings} sims')
        else:
            print(f'\nTook {time_hour:.0f} hour and {time_sec} sec')
