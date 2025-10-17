import numpy as np
import sys
from mpi4py import MPI 

def distMPI(total_tasks):
    """
    This is identical to orphics.mpi_distribute. RIP.

    Parameters
    ----------
    total_tasks : 1D list
        List-like object of all of the tasks to break up among the ranks.

    Returns
    -------
    comm, current_rank, subset
        The names are self-explanatory except for 'subset', which is the subset of tasks for the particular rank that called the function.
    """    
    #Basics
    comm = MPI.COMM_WORLD
    current_rank = comm.Get_rank()
    Nranks = comm.Get_size()

    #Break Up Work
    if len(total_tasks) == 1:
        subset = total_tasks
    else:
        tasks_per_rank, displacements = calcMPI(len(total_tasks), Nranks)
        subset = total_tasks[displacements[current_rank] : displacements[current_rank+1]]

    return comm, current_rank, subset



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



def Gatherv(comm, rank_data, Ntasks_tot, root_rank=0):
    """
    Wrapper for mpi4py's Gatherv. The main advantage is that this creates the receiving buffer for each rank, including the data type and full buffer size (taking into account the size of each array that each rank computed). This also works with distMPI, without requiring the user to call calcMPI directly.

    Parameters
    ----------
    comm : communicator object
        MPI communicator
    rank_data : ndarray
        Buffer that you're gathering for this rank
    Ntasks_tot : int
        Total number of tasks across all of your ranks combined

    Returns
    -------
    (Ntasks_tot, rank_data.shape) array
        Buffer returned by comm.Gatherv

    Raises
    ------
    NotImplementedError
        This only works for certain data types
    """
    #Get Rank Information
    Nranks = comm.Get_size()
    current_rank = comm.Get_rank()

    #Get Send Buffer Information
    indiv_array = rank_data[0]
    indiv_array_shape = list(indiv_array.shape)
    indiv_array_dtype = indiv_array.dtype
    
    #Set MPI Data Types
    if indiv_array_dtype == np.cdouble:
        mpi_dtype = MPI.DOUBLE_COMPLEX
    elif indiv_array_dtype == np.double:
        mpi_dtype = MPI.DOUBLE
    elif indiv_array_dtype == np.int_:
        mpi_dtype = MPI.LONG
    elif indiv_array_dtype == np.intc:
        mpi_dtype = MPI.INT
    else:
        raise NotImplementedError(f"Unimplemented translation between {indiv_array_dtype} and MPI's datatypes")

    #Create Receiving Buffer
    if current_rank == root_rank:
        recevbuff = np.empty( [Ntasks_tot] + indiv_array_shape , dtype= indiv_array_dtype)
    else:
        recevbuff = None

    #Get Receiving Buffer's Memory Info
    tasks_per_rank, task_displacements = calcMPI(Ntasks_tot, Nranks)
    displacements = np.prod(indiv_array_shape) * task_displacements[:-1]
    counts = np.prod(indiv_array_shape) * tasks_per_rank

    #Gatherv
    comm.Gatherv(rank_data, [recevbuff, counts, displacements, mpi_dtype], root= root_rank)

    return recevbuff



def nanReduce(comm, array, op='mean', root_rank=0):
    """Just a "wrapper" for MPI Reduce (doesn't actually call MPI's Reduce) but ignores NaN's. Supports several operations.

    Parameters
    ----------
    comm : MPI_Comm
        MPI communicator
    array : float
        Array to collect. Must be a numpy array
    op : str, optional
        Operation to be performed when combining arrays. One of ['mean', 'sum', 'min', 'max']. By default, 'mean'.
    root_rank : int, optional
        Main rank that does the assembling. By default 0.

    Returns
    -------
    array or None
        The reduced numpy array if called on the main rank; None otherwise
    """
    #Get Rank Information
    Nranks = comm.Get_size()
    current_rank = comm.Get_rank()

    #Initialize Receiving Buffer 
    recbuf = None
    if current_rank == root_rank:
        tot_shape = [Nranks] + list(array.shape)
        recbuf = np.zeros(tot_shape, dtype= array.dtype)

    comm.Gather(array, recbuf, root= root_rank)

    #Reduce Operation
    if current_rank == root_rank:
        if op == 'mean':
            return np.nanmean(recbuf, axis=0)
        elif op == 'sum':
            return np.nansum(recbuf, axis=0)
        elif op == 'min':
            return np.nanmin(recbuf, axis=0)
        elif op == 'max':
            return np.nanmax(recbuf, axis=0)
        else:
            raise NotImplementedError("Unimplemented operation")
    else:
        return None



def printSimTime(start, end, rank, simnum, rank_Nsims, tot_time, thing_type='sim'):
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
    thing_type : str, optional
        Name of the type of task (singular), by default 'sims'

    Returns
    -------
    float
        Updated total time (doesn't take into account the time it took to call and run this function)
    """
    sim_time = end - start 
    tot_time += sim_time

    print(f'\nRank {rank} completed {thing_type} {simnum} / {rank_Nsims} in {sim_time:.0f} seconds.')
    sys.stdout.flush()

    if simnum != rank_Nsims:
        remaining_time = tot_time * rank_Nsims/simnum - tot_time
        remain_min, remain_sec = divmod(remaining_time, 60)
        remain_sec = round(remain_sec)

        print(f'Approximately {remain_min :.0f} minutes and {remain_sec} seconds remaining.')
        sys.stdout.flush()

    return tot_time


def printTotalTime(start, end, hourFlag=False, Nthings=0, thing_type='sims'):
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
    thing_type : str, optional
        Name of the type of task (plural), by default 'sims'
    """

    time_min, time_sec = divmod(end-start, 60)
    
    if not hourFlag:
        #Cleanup
        time_sec = round(time_sec) 

        #Print
        if Nthings > 0:
            mpiprint(f'\nTook {time_min:.0f} min and {time_sec} sec for {Nthings} ' + thing_type)
        else:
            mpiprint(f'\nTook {time_min:.0f} min and {time_sec} sec')

    elif hourFlag:
        #Calculations
        time_hour, time_min = divmod(time_min, 60)
        time_min = round(time_min) 

        #Print
        if Nthings > 0:
            mpiprint(f'\nTook {time_hour:.0f} hour and {time_sec} sec for {Nthings} ' + thing_type)
        else:
            mpiprint(f'\nTook {time_hour:.0f} hour and {time_sec} sec')


def mpiprint(string):
    print(string)
    sys.stdout.flush()
