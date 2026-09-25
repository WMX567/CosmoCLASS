"""
.. module:: classy
    :synopsis: Python wrapper around CLASS
.. moduleauthor:: Karim Benabed <benabed@iap.fr>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
.. moduleauthor:: Julien Lesgourgues <lesgourg@cern.ch>

This module defines a class called Class. It is used with Monte Python to
extract cosmological parameters.

# JL 14.06.2017: TODO: check whether we should free somewhere the allocated fc.filename and titles, data (4 times)

"""
from math import exp,log,sqrt,pi
import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport *
cimport cython
from scipy import interpolate

ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i

# Import the .pxd containing definitions
from cclassy cimport *

DEF _MAXTITLESTRINGLENGTH_ = 8000

# Implement a specific Exception (this might not be optimally designed, nor
# even acceptable for python standards. It, however, does the job).
# The idea is to raise either an AttributeError if the problem happened while
# reading the parameters (in the normal Class, this would just return a line in
# the unused_parameters file), or a NameError in other cases. This allows
# MontePython to handle things differently.
class CosmoError(Exception):
    def __init__(self, message=""):
        self.message = message.decode() if isinstance(message,bytes) else message

    def __str__(self):
        return '\n\nError in Class: ' + self.message


class CosmoSevereError(CosmoError):
    """
    Raised when Class failed to understand one or more input parameters.

    This case would not raise any problem in Class default behaviour. However,
    for parameter extraction, one has to be sure that all input parameters were
    understood, otherwise the wrong cosmological model would be selected.
    """
    pass


class CosmoComputationError(CosmoError):
    """
    Raised when Class could not compute the cosmology at this point.

    This will be caught by the parameter extraction code to give an extremely
    unlikely value to this point
    """
    pass


cdef class Class:
    """
    Class wrapping, creates the glue between C and python

    The actual Class wrapping, the only class we will call from MontePython
    (indeed the only one we will import, with the command:
    from classy import Class

    """
    # List of used structures, defined in the header file. They have to be
    # "cdefined", because they correspond to C structures
    cdef precision pr
    cdef background ba
    cdef thermo th
    cdef perturbs pt
    cdef primordial pm
    cdef nonlinear nl
    cdef nonlinear_pt nlpt
    cdef transfers tr
    cdef spectra sp
    cdef output op
    cdef lensing le
    cdef file_content fc

    cdef int ready # Flag
    cdef object _pars # Dictionary of the parameters
    cdef object ncp   # Keeps track of the structures initialized, in view of cleaning.

    cdef np.ndarray pk_mult
    cdef np.ndarray kh
    cdef double fz
    cdef int output_init

    # Defining two new properties to recover, respectively, the parameters used
    # or the age (set after computation). Follow this syntax if you want to
    # access other quantities. Alternatively, you can also define a method, and
    # call it (see _T_cmb method, at the very bottom).
    property pars:
        def __get__(self):
            return self._pars
    property state:
        def __get__(self):
            return self.ready
    property Omega_nu:
        def __get__(self):
            return self.ba.Omega0_ncdm_tot
    property nonlinear_method:
        def __get__(self):
            return self.nl.method
    property nonlinear_pt_method:
        def __get__(self):
            return self.nlpt.method

    def set_default(self):
        _pars = {
            "output":"tCl mPk",}
        self.set(**_pars)

    def __cinit__(self, default=False):
        cdef char* dumc
        self.ready = False
        self.output_init = False
        self._pars = {}
        self.fc.size=0
        self.fc.filename = <char*>malloc(sizeof(char)*30)
        assert(self.fc.filename!=NULL)
        dumc = "NOFILE"
        sprintf(self.fc.filename,"%s",dumc)
        self.ncp = set()
        if default: self.set_default()

    # Set up the dictionary
    def set(self,*pars,**kars):
        if len(pars)==1:
            self._pars.update(dict(pars[0]))
        elif len(pars)!=0:
            raise CosmoSevereError("bad call")
        self._pars.update(kars)
        self.ready=False
        return True

    def empty(self):
        self._pars = {}
        self.ready = False

    # Create an equivalent of the parameter file. Non specified values will be
    # taken at their default (in Class)
    def _fillparfile(self):
        cdef char* dumc

        if self.fc.size!=0:
            free(self.fc.name)
            free(self.fc.value)
            free(self.fc.read)
        self.fc.size = len(self._pars)
        self.fc.name = <FileArg*> malloc(sizeof(FileArg)*len(self._pars))
        assert(self.fc.name!=NULL)

        self.fc.value = <FileArg*> malloc(sizeof(FileArg)*len(self._pars))
        assert(self.fc.value!=NULL)

        self.fc.read = <short*> malloc(sizeof(short)*len(self._pars))
        assert(self.fc.read!=NULL)

        # fill parameter file
        i = 0
        for kk in self._pars:

            dumcp = kk.encode()
            dumc = dumcp
            sprintf(self.fc.name[i],"%s",dumc)
            dumcp = str(self._pars[kk]).encode()
            dumc = dumcp
            sprintf(self.fc.value[i],"%s",dumc)
            self.fc.read[i] = _FALSE_
            i+=1

    # Called at the end of a run, to free memory
    def struct_cleanup(self):
        if self.ready == _FALSE_:
             return
        if "lensing" in self.ncp:
            lensing_free(&self.le)
        if "spectra" in self.ncp:
            spectra_free(&self.sp)
        if "transfer" in self.ncp:
            transfer_free(&self.tr)
        if "nonlinear" in self.ncp:
            nonlinear_free(&self.nl)
        if "nonlinear_pt" in self.ncp:
            nonlinear_pt_free(&self.nlpt)
        if "primordial" in self.ncp:
            primordial_free(&self.pm)
        if "perturb" in self.ncp:
            perturb_free(&self.pt)
        if "thermodynamics" in self.ncp:
            thermodynamics_free(&self.th)
        if "background" in self.ncp:
            background_free(&self.ba)
        self.ready = False

    def _check_task_dependency(self, level):
        """
        Fill the level list with all the needed modules

        .. warning::

            the ordering of modules is obviously dependent on CLASS module order
            in the main.c file. This has to be updated in case of a change to
            this file.

        Parameters
        ----------

        level : list
            list of strings, containing initially only the last module required.
            For instance, to recover all the modules, the input should be
            ['lensing']

        """
        if "lensing" in level:
            level.append("spectra")
        if "spectra" in level:
            level.append("transfer")
        if "transfer" in level:
            level.append("nonlinear_pt")
        if "nonlinear_pt" in level:
            level.append("nonlinear")
        if "nonlinear" in level:
            level.append("primordial")
        if "primordial" in level:
            level.append("perturb")
        if "perturb" in level:
            level.append("thermodynamics")
        if "thermodynamics" in level:
            level.append("background")
        if len(level)!=0 :
            level.append("input")
        return level

    def _pars_check(self, key, value, contains=False, add=""):
        val = ""
        if key in self._pars:
            val = self._pars[key]
            if contains:
                if value in val:
                    return True
            else:
                if value==val:
                    return True
        if add:
            sep = " "
            if isinstance(add,str):
                sep = add

            if contains and val:
                    self.set({key:val+sep+value})
            else:
                self.set({key:value})
            return True
        return False

    def compute(self, level=["lensing"]):
        """
        compute(level=["lensing"])

        Main function, execute all the _init methods for all desired modules.
        This is called in MontePython, and this ensures that the Class instance
        of this class contains all the relevant quantities. Then, one can deduce
        Pk, Cl, etc...

        Parameters
        ----------
        level : list
                list of the last module desired. The internal function
                _check_task_dependency will then add to this list all the
                necessary modules to compute in order to initialize this last
                one. The default last module is "lensing".

        .. warning::

            level default value should be left as an array (it was creating
            problem when casting as a set later on, in _check_task_dependency)

        """
        cdef ErrorMsg errmsg

        # Append to the list level all the modules necessary to compute.
        level = self._check_task_dependency(level)

        # Check if this function ran before (self.ready should be true), and
        # if no other modules were requested, i.e. if self.ncp contains (or is
        # equivalent to) level. If it is the case, simply stop the execution of
        # the function.
        if self.ready and self.ncp.issuperset(level):
            return

        # Otherwise, proceed with the normal computation.
        self.ready = False

        # Equivalent of writing a parameter file
        self._fillparfile()

        # self.ncp will contain the list of computed modules (under the form of
        # a set, instead of a python list)
        self.ncp=set()

        # --------------------------------------------------------------------
        # Check the presence for all CLASS modules in the list 'level'. If a
        # module is found in level, executure its "_init" method.
        # --------------------------------------------------------------------
        # The input module should raise a CosmoSevereError, because
        # non-understood parameters asked to the wrapper is a problematic
        # situation.
        if "input" in level:
            if input_init(&self.fc, &self.pr, &self.ba, &self.th,
                          &self.pt, &self.tr, &self.pm, &self.sp,
                          &self.nl, &self.nlpt, &self.le, &self.op, errmsg) == _FAILURE_:
                raise CosmoSevereError(errmsg)
            self.ncp.add("input")
            # This part is done to list all the unread parameters, for debugging
            problem_flag = False
            problematic_parameters = []
            for i in range(self.fc.size):
                if self.fc.read[i] == _FALSE_:
                    problem_flag = True
                    problematic_parameters.append(self.fc.name[i].decode())
            if problem_flag:
                raise CosmoSevereError(
                    "Class did not read input parameter(s): %s\n" % ', '.join(
                    problematic_parameters))

        # The following list of computation is straightforward. If the "_init"
        # methods fail, call `struct_cleanup` and raise a CosmoComputationError
        # with the error message from the faulty module of CLASS.
        if "background" in level:
            if background_init(&(self.pr), &(self.ba)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.ba.error_message)
            self.ncp.add("background")

        if "thermodynamics" in level:
            if thermodynamics_init(&(self.pr), &(self.ba),
                                   &(self.th)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.th.error_message)
            self.ncp.add("thermodynamics")

        if "perturb" in level:
            if perturb_init(&(self.pr), &(self.ba),
                            &(self.th), &(self.pt)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.pt.error_message)
            self.ncp.add("perturb")

        if "primordial" in level:
            if primordial_init(&(self.pr), &(self.pt),
                               &(self.pm)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.pm.error_message)
            self.ncp.add("primordial")

        if "nonlinear" in level:
            if nonlinear_init(&self.pr, &self.ba, &self.th,
                              &self.pt, &self.pm, &self.nl) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.nl.error_message)
            self.ncp.add("nonlinear")

        if "nonlinear_pt" in level:
            if nonlinear_pt_init(&self.pr, &self.ba, &self.th,
                              &self.pt, &self.pm, &self.nlpt) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.nlpt.error_message)
            self.ncp.add("nonlinear_pt")

        if "transfer" in level:
            if transfer_init(&(self.pr), &(self.ba), &(self.th),
                             &(self.pt), &(self.nlpt), &(self.nl), &(self.tr)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.tr.error_message)
            self.ncp.add("transfer")

        if "spectra" in level:
            if spectra_init(&(self.pr), &(self.ba), &(self.pt),
                            &(self.pm), &(self.nlpt), &(self.nl), &(self.tr),
                            &(self.sp)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.sp.error_message)
            self.ncp.add("spectra")

        if "lensing" in level:
            if lensing_init(&(self.pr), &(self.pt), &(self.sp),
                            &(self.nlpt), &(self.nl), &(self.le)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.le.error_message)
            self.ncp.add("lensing")

        self.ready = True

        # At this point, the cosmological instance contains everything needed. The
        # following functions are only to output the desired numbers
        return

    def recompute_nonlinear_pt(self, no_wiggle=False, alpha_rs=1.0, force=False):
        recompute=False
        if no_wiggle:
            if self.nlpt.no_wiggle == _FALSE_:
                recompute=True
            self.nlpt.no_wiggle=_TRUE_
        else:
            if self.nlpt.no_wiggle == _TRUE_:
                recompute=True
            self.nlpt.no_wiggle=_FALSE_

        if self.nlpt.alpha_rs != alpha_rs:
            self.nlpt.alpha_rs =  alpha_rs
            recompute = True

        if (recompute or force) and self.ready:
            if nonlinear_pt_init(&self.pr, &self.ba, &self.th,&self.pt, &self.pm, &self.nlpt) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.nlpt.error_message)
            self.ncp.add("nonlinear_pt")

    def raw_cl(self, lmax=-1, nofail=False):
        """
        raw_cl(lmax=-1, nofail=False)

        Return a dictionary of the primary C_l

        Parameters
        ----------
        lmax : int, optional
                Define the maximum l for which the C_l will be returned
                (inclusively). This number will be checked against the maximum l
                at which they were actually computed by CLASS, and an error will
                be raised if the desired lmax is bigger than what CLASS can
                give.
        nofail: bool, optional
                Check and enforce the computation of the spectra module
                beforehand, with the desired lmax.

        Returns
        -------
        cl : dict
                Dictionary that contains the power spectrum for 'tt', 'te', etc... The
                index associated with each is defined wrt. Class convention, and are non
                important from the python point of view. It also returns now the
                ell array.
        """
        cdef int lmaxR
        cdef double *rcl = <double*> calloc(self.sp.ct_size,sizeof(double))

        # Quantities for tensor modes
        cdef double **cl_md = <double**> calloc(self.sp.md_size, sizeof(double*))
        for index_md in range(self.sp.md_size):
            cl_md[index_md] = <double*> calloc(self.sp.ct_size, sizeof(double))

        # Quantities for isocurvature modes
        cdef double **cl_md_ic = <double**> calloc(self.sp.md_size, sizeof(double*))
        for index_md in range(self.sp.md_size):
            cl_md_ic[index_md] = <double*> calloc(self.sp.ct_size*self.sp.ic_ic_size[index_md], sizeof(double))

        # Define a list of integers, refering to the flags and indices of each
        # possible output Cl. It allows for a clear and concise way of looping
        # over them, checking if they are defined or not.
        has_flags = [
            (self.sp.has_tt, self.sp.index_ct_tt, 'tt'),
            (self.sp.has_ee, self.sp.index_ct_ee, 'ee'),
            (self.sp.has_te, self.sp.index_ct_te, 'te'),
            (self.sp.has_bb, self.sp.index_ct_bb, 'bb'),
            (self.sp.has_pp, self.sp.index_ct_pp, 'pp'),
            (self.sp.has_tp, self.sp.index_ct_tp, 'tp'),]
        spectra = []

        for flag, index, name in has_flags:
            if flag:
                spectra.append(name)

        if not spectra:
            raise CosmoSevereError("No Cl computed")
        lmaxR = self.sp.l_max_tot
        if lmax == -1:
            lmax = lmaxR
        if lmax > lmaxR:
            if nofail:
                self._pars_check("l_max_scalars",lmax)
                self.compute(["lensing"])
            else:
                raise CosmoSevereError("Can only compute up to lmax=%d"%lmaxR)

        # Initialise all the needed Cls arrays
        cl = {}
        for elem in spectra:
            cl[elem] = np.zeros(lmax+1, dtype=np.double)

        # Recover for each ell the information from CLASS
        for ell from 2<=ell<lmax+1:
            if spectra_cl_at_l(&self.sp, ell, rcl, cl_md, cl_md_ic) == _FAILURE_:
                raise CosmoSevereError(self.sp.error_message)
            for flag, index, name in has_flags:
                if name in spectra:
                    cl[name][ell] = rcl[index]
        cl['ell'] = np.arange(lmax+1)

        free(rcl)
        for index_md in range(self.sp.md_size):
            free(cl_md[index_md])
            free(cl_md_ic[index_md])
        free(cl_md)
        free(cl_md_ic)

        return cl

    def lensed_cl(self, lmax=-1,nofail=False):
        """
        lensed_cl(lmax=-1, nofail=False)

        Return a dictionary of the lensed C_l, computed by CLASS, without the
        density C_ls. They must be asked separately with the function aptly
        named density_cl

        Parameters
        ----------
        lmax : int, optional
                Define the maximum l for which the C_l will be returned (inclusively)
        nofail: bool, optional
                Check and enforce the computation of the lensing module beforehand

        Returns
        -------
        cl : dict
                Dictionary that contains the power spectrum for 'tt', 'te', etc... The
                index associated with each is defined wrt. Class convention, and are non
                important from the python point of view.
        """
        cdef int lmaxR
        cdef double *lcl = <double*> calloc(self.le.lt_size,sizeof(double))

        # Define a list of integers, refering to the flags and indices of each
        # possible output Cl. It allows for a clear and concise way of looping
        # over them, checking if they are defined or not.
        has_flags = [
            (self.le.has_tt, self.le.index_lt_tt, 'tt'),
            (self.le.has_ee, self.le.index_lt_ee, 'ee'),
            (self.le.has_te, self.le.index_lt_te, 'te'),
            (self.le.has_bb, self.le.index_lt_bb, 'bb'),
            (self.le.has_pp, self.le.index_lt_pp, 'pp'),
            (self.le.has_tp, self.le.index_lt_tp, 'tp'),]
        spectra = []

        for flag, index, name in has_flags:
            if flag:
                spectra.append(name)

        if not spectra:
            raise CosmoSevereError("No lensed Cl computed")
        lmaxR = self.le.l_lensed_max

        if lmax == -1:
            lmax = lmaxR
        if lmax > lmaxR:
            if nofail:
                self._pars_check("l_max_scalars",lmax)
                self.compute(["lensing"])
            else:
                raise CosmoSevereError("Can only compute up to lmax=%d"%lmaxR)

        cl = {}
        # Simple Cls, for temperature and polarisation, are not so big in size
        for elem in spectra:
            cl[elem] = np.zeros(lmax+1, dtype=np.double)
        for ell from 2<=ell<lmax+1:
            if lensing_cl_at_l(&self.le,ell,lcl) == _FAILURE_:
                raise CosmoSevereError(self.le.error_message)
            for flag, index, name in has_flags:
                if name in spectra:
                    cl[name][ell] = lcl[index]
        cl['ell'] = np.arange(lmax+1)

        free(lcl)
        return cl

    def density_cl(self, lmax=-1, nofail=False):
        """
        density_cl(lmax=-1, nofail=False)

        Return a dictionary of the primary C_l for the matter

        Parameters
        ----------
        lmax : int, optional
            Define the maximum l for which the C_l will be returned (inclusively)
        nofail: bool, optional
            Check and enforce the computation of the lensing module beforehand

        Returns
        -------
        cl : numpy array of numpy.ndarrays
            Array that contains the list (in this order) of self correlation of
            1st bin, then successive correlations (set by non_diagonal) to the
            following bins, then self correlation of 2nd bin, etc. The array
            starts at index_ct_dd.
        """
        cdef int lmaxR
        cdef double *dcl = <double*> calloc(self.sp.ct_size,sizeof(double))

        # Quantities for tensor modes
        cdef double **cl_md = <double**> calloc(self.sp.md_size, sizeof(double*))
        for index_md in range(self.sp.md_size):
            cl_md[index_md] = <double*> calloc(self.sp.ct_size, sizeof(double))

        # Quantities for isocurvature modes
        cdef double **cl_md_ic = <double**> calloc(self.sp.md_size, sizeof(double*))
        for index_md in range(self.sp.md_size):
            cl_md_ic[index_md] = <double*> calloc(self.sp.ct_size*self.sp.ic_ic_size[index_md], sizeof(double))

        lmaxR = self.pt.l_lss_max
        has_flags = [
            (self.sp.has_dd, self.sp.index_ct_dd, 'dd'),
            (self.sp.has_td, self.sp.index_ct_td, 'td'),
            (self.sp.has_ll, self.sp.index_ct_ll, 'll'),
            (self.sp.has_dl, self.sp.index_ct_dl, 'dl'),
            (self.sp.has_tl, self.sp.index_ct_tl, 'tl')]
        spectra = []

        for flag, index, name in has_flags:
            if flag:
                spectra.append(name)
                l_max_flag = self.sp.l_max_ct[self.sp.index_md_scalars][index]
                if l_max_flag < lmax and lmax > 0:
                    raise CosmoSevereError(
                        "the %s spectrum was computed until l=%i " % (
                            name.upper(), l_max_flag) +
                        "but you asked a l=%i" % lmax)

        if not spectra:
            raise CosmoSevereError("No density Cl computed")
        if lmax == -1:
            lmax = lmaxR
        if lmax > lmaxR:
            if nofail:
                self._pars_check("l_max_lss",lmax)
                self._pars_check("output",'nCl')
                self.compute()
            else:
                raise CosmoSevereError("Can only compute up to lmax=%d"%lmaxR)

        cl = {}

        # For density Cls, the size is bigger (different redshfit bins)
        # computes the size, given the number of correlations needed to be computed
        size = (self.sp.d_size*(self.sp.d_size+1)-(self.sp.d_size-self.sp.non_diag)*
                (self.sp.d_size-1-self.sp.non_diag))/2;
        for elem in ['dd', 'll', 'dl']:
            if elem in spectra:
                cl[elem] = {}
                for index in range(size):
                    cl[elem][index] = np.zeros(
                        lmax+1, dtype=np.double)
        for elem in ['td', 'tl']:
            if elem in spectra:
                cl[elem] = np.zeros(lmax+1, dtype=np.double)

        for ell from 2<=ell<lmax+1:
            if spectra_cl_at_l(&self.sp, ell, dcl, cl_md, cl_md_ic) == _FAILURE_:
                raise CosmoSevereError(self.sp.error_message)
            if 'dd' in spectra:
                for index in range(size):
                    cl['dd'][index][ell] = dcl[self.sp.index_ct_dd+index]
            if 'll' in spectra:
                for index in range(size):
                    cl['ll'][index][ell] = dcl[self.sp.index_ct_ll+index]
            if 'dl' in spectra:
                for index in range(size):
                    cl['dl'][index][ell] = dcl[self.sp.index_ct_dl+index]
            if 'td' in spectra:
                cl['td'][ell] = dcl[self.sp.index_ct_td]
            if 'tl' in spectra:
                cl['tl'][ell] = dcl[self.sp.index_ct_tl]
        cl['ell'] = np.arange(lmax+1)

        free(dcl)
        for index_md in range(self.sp.md_size):
            free(cl_md[index_md])
            free(cl_md_ic[index_md])
        free(cl_md)
        free(cl_md_ic)

        return cl

    def z_of_r (self,z_array):
        cdef double tau=0.0
        cdef int last_index=0 #junk
        cdef double * pvecback
        r = np.zeros(len(z_array),'float64')
        dzdr = np.zeros(len(z_array),'float64')

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        i = 0
        for redshift in z_array:
            if background_tau_of_z(&self.ba,redshift,&tau)==_FAILURE_:
                raise CosmoSevereError(self.ba.error_message)

            if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
                raise CosmoSevereError(self.ba.error_message)

            # store r
            r[i] = pvecback[self.ba.index_bg_conf_distance]
            # store dz/dr = H
            dzdr[i] = pvecback[self.ba.index_bg_H]

            i += 1

        free(pvecback)
        return r[:],dzdr[:]

    def luminosity_distance(self, z):
        """
        luminosity_distance(z)
        """
        cdef double tau=0.0
        cdef int last_index = 0  # junk
        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        if background_tau_of_z(&self.ba, z, &tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba, tau, self.ba.long_info,
                self.ba.inter_normal, &last_index, pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)
        lum_distance = pvecback[self.ba.index_bg_lum_distance]
        free(pvecback)
        return lum_distance

    # Gives the pk for a given (k,z)
    def pk(self,double k,double z, int no_wiggle=False, double alpha_rs=1.0):

        """
        Gives the pk for a given k and z (will be non linear if requested to Class, linear otherwise)

        .. note::

            there is an additional check to verify if output contains `mPk`,
            because otherwise a segfault will occur

        """
        cdef double pk #--> then this counts as well!!!
        cdef double pk_Id2d2 #1
        cdef double pk_Id2 #2
        cdef double pk_IG2 #2
        cdef double pk_Id2G2 #4
        cdef double pk_IG2G2 #5
        cdef double pk_IFG2 #6
        cdef double pk_IFG2_0b1 #7
        cdef double pk_IFG2_0 #8
        cdef double pk_IFG2_2 #9
        cdef double pk_CTR #10
        cdef double pk_CTR_0 #11
        cdef double pk_CTR_2 #12
        cdef double pk_CTR_4 #13
        cdef double pk_Tree #14
        cdef double pk_Tree_0_vv #15
        cdef double pk_Tree_0_vd #16
        cdef double pk_Tree_0_dd #17
        cdef double pk_Tree_2_vv #18
        cdef double pk_Tree_2_vd #19
        cdef double pk_Tree_4_vv #20
        cdef double pk_0_vv #21
        cdef double pk_0_vd #22
        cdef double pk_0_dd #23
        cdef double pk_2_vv #24
        cdef double pk_2_vd #25
        cdef double pk_2_dd #26
        cdef double pk_4_vv #27
        cdef double pk_4_vd #28
        cdef double pk_4_dd #29
        cdef double pk_0_b1b2 #30
        cdef double pk_0_b2 #31
        cdef double pk_0_b1bG2 #32
        cdef double pk_0_bG2 #33
        cdef double pk_2_b1b2 #34
        cdef double pk_2_b2 #35
        cdef double pk_2_b1bG2 #36
        cdef double pk_2_bG2 #37
        cdef double pk_4_b2 #38
        cdef double pk_4_bG2 #39
        cdef double pk_4_b1b2 #40
        cdef double pk_4_b1bG2 #41
        cdef double pk_2_b2b2 #42
        cdef double pk_2_b2bG2 #43
        cdef double pk_2_bG2bG2 #44
        cdef double pk_4_b2b2 #45
        cdef double pk_4_b2bG2 #46
        cdef double pk_4_bG2bG2 #47
        #GC: it doesn't seem like the order matters... This I need to understand better...
        #GC -> 47+1. Then, I have 49 to 72 because I have 48+3+9+12=72...
        cdef double pk_nl_fNL #48
        cdef double pk_fNLd2 #49
        cdef double pk_fNLG2 #50
        #GC...
        cdef double pk_fNL_0_vv #51
        cdef double pk_fNL_0_vd #52
        cdef double pk_fNL_0_dd #53
        cdef double pk_fNL_2_vv #54
        cdef double pk_fNL_2_vd #55
        cdef double pk_fNL_2_dd #56
        cdef double pk_fNL_4_vv #57
        cdef double pk_fNL_4_vd #58
        cdef double pk_fNL_4_dd #59
        #GC...
        cdef double pk12_0_b1b2 #60
        cdef double pk12_0_b2 #61
        cdef double pk12_0_b1bG2 #62
        cdef double pk12_0_bG2 #63
        cdef double pk12_2_b1b2 #64
        cdef double pk12_2_b2 #65
        cdef double pk12_2_b1bG2 #66
        cdef double pk12_2_bG2 #67
        cdef double pk12_4_b1b2 #68
        cdef double pk12_4_b2 #69
        cdef double pk12_4_b1bG2 #70
        cdef double pk12_4_bG2 #71
        #GC: ORTHOGONAL -- start
        cdef double pk_nl_fNL_ortho #72 -> 73
        cdef double pk_fNLd2_ortho #73 -> 74
        cdef double pk_fNLG2_ortho #74 -> 75
        #GC...
        cdef double pk_fNL_0_vv_ortho #75 -> 76
        cdef double pk_fNL_0_vd_ortho #76 -> 77
        cdef double pk_fNL_0_dd_ortho #77 -> 78
        cdef double pk_fNL_2_vv_ortho #78 -> 79
        cdef double pk_fNL_2_vd_ortho #79 -> 80
        cdef double pk_fNL_2_dd_ortho #80 -> 81
        cdef double pk_fNL_4_vv_ortho #81 -> 82
        cdef double pk_fNL_4_vd_ortho #82 -> 83
        cdef double pk_fNL_4_dd_ortho #83 -> 84
        #GC...
        cdef double pk12_0_b1b2_ortho #84 -> 85
        cdef double pk12_0_b2_ortho #85 -> 86
        cdef double pk12_0_b1bG2_ortho #86 -> 87
        cdef double pk12_0_bG2_ortho #87 -> 88
        cdef double pk12_2_b1b2_ortho #88 -> 89
        cdef double pk12_2_b2_ortho #89 -> 90
        cdef double pk12_2_b1bG2_ortho #90 -> 91
        cdef double pk12_2_bG2_ortho #91 -> 92
        cdef double pk12_4_b1b2_ortho #92 -> 93
        cdef double pk12_4_b2_ortho #93 -> 94
        cdef double pk12_4_b1bG2_ortho #94 -> 95
        cdef double pk12_4_bG2_ortho  #95 -> 96
        #GC: ORTHOGONAL -- finish
        #GC -> end here...
        cdef double pk_velo
        cdef double pk_cross
        cdef int dummy

        #cdef double large_for_logs_fNL = 1000. #GC...

        cdef double large_for_logs_matter = 5000.
        cdef double large_for_logs_big = 1000000.
        cdef double large_for_logs_small = 10.

        #GC: ORTHOGONAL -- start
        cdef double large_for_logs_fNL = 50000.
        #GC: ORTHOGONAL -- finish

        cdef double factor_fNL = sqrt(self.pm.A_s) * (1944./625.) * (pi**4.) #GC: multiplied already by sqrt(As) * (1944/625*pi^4), so the user only needs to multiply by fNL...

        # Quantities for the isocurvature modes
        cdef double *pk_ic = <double*> calloc(self.sp.ic_ic_size[self.sp.index_md_scalars], sizeof(double))
        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError(
                "No power spectrum computed. You must add mPk to the list of outputs."
                )

        self.recompute_nonlinear_pt(no_wiggle=no_wiggle, alpha_rs=alpha_rs)
        


        if (self.nlpt.method == 0):
             if spectra_pk_at_k_and_z(&self.ba,&self.pm,&self.sp,k,z,&pk,pk_ic)==_FAILURE_:
                raise CosmoSevereError(self.sp.error_message)
        else:
             if spectra_pk_nl_at_k_and_z(&self.ba,&self.pm,&self.sp,&self.nl,&self.nlpt,k,z,&pk,&pk_Id2d2,&pk_Id2,&pk_IG2,&pk_Id2G2,&pk_IG2G2,&pk_IFG2,&pk_IFG2_0b1,&pk_IFG2_0,&pk_IFG2_2,&pk_CTR,&pk_CTR_0,&pk_CTR_2,&pk_CTR_4,&pk_Tree,&pk_Tree_0_vv,&pk_Tree_0_vd,&pk_Tree_0_dd,&pk_Tree_2_vv,&pk_Tree_2_vd,&pk_Tree_4_vv,&pk_0_vv,&pk_0_vd,&pk_0_dd,&pk_2_vv,&pk_2_vd,&pk_2_dd,&pk_4_vv,&pk_4_vd,&pk_4_dd,&pk_0_b1b2,&pk_0_b2,&pk_0_b1bG2,&pk_0_bG2,&pk_2_b1b2,&pk_2_b2,&pk_2_b1bG2,&pk_2_bG2,&pk_4_b2,&pk_4_bG2,&pk_4_b1b2,&pk_4_b1bG2,&pk_2_b2b2,&pk_2_b2bG2,&pk_2_bG2bG2,&pk_4_b2b2,&pk_4_b2bG2,&pk_4_bG2bG2,&pk_nl_fNL,&pk_fNLd2,&pk_fNLG2,&pk_fNL_0_vv,&pk_fNL_0_vd,&pk_fNL_0_dd,&pk_fNL_2_vv,&pk_fNL_2_vd,&pk_fNL_2_dd,&pk_fNL_4_vv,&pk_fNL_4_vd,&pk_fNL_4_dd,&pk12_0_b1b2,&pk12_0_b2,&pk12_0_b1bG2,&pk12_0_bG2,&pk12_2_b1b2,&pk12_2_b2,&pk12_2_b1bG2,&pk12_2_bG2,&pk12_4_b1b2,&pk12_4_b2,&pk12_4_b1bG2,&pk12_4_bG2,&pk_nl_fNL_ortho,&pk_fNLd2_ortho,&pk_fNLG2_ortho,&pk_fNL_0_vv_ortho,&pk_fNL_0_vd_ortho,&pk_fNL_0_dd_ortho,&pk_fNL_2_vv_ortho,&pk_fNL_2_vd_ortho,&pk_fNL_2_dd_ortho,&pk_fNL_4_vv_ortho,&pk_fNL_4_vd_ortho,&pk_fNL_4_dd_ortho,&pk12_0_b1b2_ortho,&pk12_0_b2_ortho,&pk12_0_b1bG2_ortho,&pk12_0_bG2_ortho,&pk12_2_b1b2_ortho,&pk12_2_b2_ortho,&pk12_2_b1bG2_ortho,&pk12_2_bG2_ortho,&pk12_4_b1b2_ortho,&pk12_4_b2_ortho,&pk12_4_b1bG2_ortho,&pk12_4_bG2_ortho) ==_FAILURE_:
                raise CosmoSevereError(self.sp.error_message)
        result = [pk-large_for_logs_matter]
        result.append(-pk_Id2d2+large_for_logs_big)
        result.append(pk_Id2-large_for_logs_small)
        result.append(-pk_IG2+large_for_logs_small)
        result.append(-pk_Id2G2+large_for_logs_big)
        result.append(pk_IG2G2-large_for_logs_big)
        result.append(-pk_IFG2+large_for_logs_small)
        result.append(-pk_IFG2_0b1+large_for_logs_big)
        result.append(-pk_IFG2_0+large_for_logs_big)
        result.append(-pk_IFG2_2+large_for_logs_big)
        result.append(-pk_CTR)
        result.append(-pk_CTR_0)
        result.append(-pk_CTR_2)
        result.append(-pk_CTR_4)
        result.append(pk_Tree)
        result.append(pk_Tree_0_vv-large_for_logs_big)
        result.append(pk_Tree_0_vd-large_for_logs_big)
        result.append(pk_Tree_0_dd-large_for_logs_big)
        result.append(pk_Tree_2_vv-large_for_logs_big)
        result.append(pk_Tree_2_vd-large_for_logs_big)
        result.append(pk_Tree_4_vv-large_for_logs_big)
        result.append(pk_0_vv-large_for_logs_big)
        result.append(pk_0_vd-large_for_logs_big)
        result.append(pk_0_dd-large_for_logs_big)
        result.append(pk_2_vv-large_for_logs_big)
        result.append(pk_2_vd-large_for_logs_big)
        result.append(pk_2_dd-large_for_logs_big)
        result.append(pk_4_vv-large_for_logs_big)
        result.append(pk_4_vd-large_for_logs_big)
        result.append(pk_4_dd-large_for_logs_big)
        result.append(pk_0_b1b2-large_for_logs_big)
        result.append(pk_0_b2-large_for_logs_big)
        result.append(-pk_0_b1bG2+large_for_logs_big)
        result.append(-pk_0_bG2+large_for_logs_big)
        result.append(pk_2_b1b2-large_for_logs_big)
        result.append(pk_2_b2-large_for_logs_big)
        result.append(-pk_2_b1bG2+large_for_logs_big)
        result.append(-pk_2_bG2+large_for_logs_big)
        result.append(pk_4_b2-large_for_logs_big)
        result.append(-pk_4_bG2+large_for_logs_big)
        result.append(pk_4_b1b2-large_for_logs_big)
        result.append(pk_4_b1bG2-large_for_logs_big)
        result.append(pk_2_b2b2-large_for_logs_big)
        result.append(pk_2_b2bG2-large_for_logs_big)
        result.append(pk_2_bG2bG2-large_for_logs_big)
        result.append(pk_4_b2b2-large_for_logs_big)
        result.append(pk_4_b2bG2-large_for_logs_big)
        result.append(pk_4_bG2bG2-large_for_logs_big)
        #GC...
        result.append(factor_fNL*(pk_nl_fNL - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNLd2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNLG2 - large_for_logs_fNL))
        #GC...
        result.append(factor_fNL*(pk_fNL_0_vv - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_0_vd - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_0_dd - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_2_vv - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_2_vd - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_2_dd - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_4_vv - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_4_vd - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_4_dd - large_for_logs_fNL))
        #GC...
        result.append(factor_fNL*(pk12_0_b1b2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_0_b2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_0_b1bG2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_0_bG2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_b1b2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_b2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_b1bG2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_bG2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_b1b2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_b2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_b1bG2 - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_bG2 - large_for_logs_fNL))
        #GC: ORTHOGONAL -- start
        #GC...
        result.append(factor_fNL*(pk_nl_fNL_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNLd2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNLG2_ortho - large_for_logs_fNL))
        #GC...
        result.append(factor_fNL*(pk_fNL_0_vv_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_0_vd_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_0_dd_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_2_vv_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_2_vd_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_2_dd_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_4_vv_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_4_vd_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk_fNL_4_dd_ortho - large_for_logs_fNL))
        #GC...
        result.append(factor_fNL*(pk12_0_b1b2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_0_b2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_0_b1bG2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_0_bG2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_b1b2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_b2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_b1bG2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_2_bG2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_b1b2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_b2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_b1bG2_ortho - large_for_logs_fNL))
        result.append(factor_fNL*(pk12_4_bG2_ortho - large_for_logs_fNL))
        #GC: ORTHOGONAL -- finish
        free(pk_ic)
#        return pk
        return result

    def pk_halofit(self, double k, double z):
        """Total matter Halofit P(k,z), in Mpc^3 with k in 1/Mpc."""
        cdef double pk
        self.compute()
        if self.pt.has_pk_matter == _FALSE_ or self.nl.method == 0:
            raise CosmoSevereError("pk_halofit requires output=mPk and non linear=halofit")
        if spectra_pk_nl_halofit_at_k_and_z(&self.ba, &self.pm, &self.sp,
                                           k, z, &pk) == _FAILURE_:
            raise CosmoSevereError(self.sp.error_message)
        return pk

    # Gives the linear pk for a given (k,z)
    def pk_lin(self,double k,double z):
        """
        Gives the linear pk for a given k and z (even if non linear corrections were requested to Class)

        .. note::

            there is an additional check to verify if output contains `mPk`,
            because otherwise a segfault will occur

        """
        cdef double pk
        cdef double pk_velo
        cdef double pk_cross
        cdef int dummy

        # Quantities for the isocurvature modes
        cdef double *pk_ic = <double*> calloc(self.sp.ic_ic_size[self.sp.index_md_scalars], sizeof(double))
        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError(
                "No power spectrum computed. You must add mPk to the list of outputs."
                )

        if spectra_pk_at_k_and_z(&self.ba,&self.pm,&self.sp,k,z,&pk,pk_ic)==_FAILURE_:
            raise CosmoSevereError(self.sp.error_message)

        free(pk_ic)
        return pk

    def get_pk(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """ Fast function to get the power spectrum on a k and z array """
        cdef np.ndarray[DTYPE_t, ndim=3] pk = np.zeros((k_size,z_size,mu_size),'float64')
        cdef int index_k, index_z, index_mu

        for index_k in xrange(k_size):
            for index_z in xrange(z_size):
                for index_mu in xrange(mu_size):
                    pk[index_k,index_z,index_mu] = self.pk(k[index_k,index_z,index_mu],z[index_z])
        return pk

    def get_pk_mult(self, np.ndarray[DTYPE_t,ndim=1] k, double z, int k_size, no_wiggle=False, alpha_rs=1.0):
        """Fast function to get the non-linear power spectrum multipole components on a k array"""
        #cdef np.ndarray[DTYPE_t, ndim = 2] pk_mult = np.zeros((72,k_size),'float64')
        #GC: ORTHOGONAL -- start
        cdef np.ndarray[DTYPE_t, ndim = 2] pk_mult = np.zeros((96,k_size),'float64')
        #GC: ORTHOGONAL -- finish
        cdef np.ndarray[DTYPE_t, ndim = 1] this_pk
        cdef int index_k, index_comb

        for index_k in xrange(k_size):
            this_pk = np.asarray(self.pk(k[index_k],z, no_wiggle=no_wiggle, alpha_rs=alpha_rs))
            for index_comb in xrange(96): #GC: ORTHOGONAL...
                pk_mult[index_comb, index_k] = this_pk[index_comb]

        return pk_mult

    def pk_1l_sel(self, double z, int Nmax = 128):


        def M13direct_0(n1, f):
            return (1 + 9*n1)/(112.*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_1(n1, f):
            return (9*f)/(56.*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1)

        def M13direct_2(n1, f):
            return (-3*f*(5 + 6*f - 3*n1))/(112.*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_3(n1, f):
            return (9*f**2)/(56.*(-3 + n1)*(-2 + n1)*(-1 + n1)*(1 + n1))

        def M13direct_4(n1, f):
            return -15/(14.*n1*(-6 + 5*n1 + 5*n1**2 - 5*n1**3 + n1**4))

        def M13direct_5(n1, f):
            return (-405*f**2)/(224.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_6(n1, f):
            return (3*f*(100 - 105*n1 + 32*n1**2 - 3*n1**3 + 9*f*(15 + 2*n1 + 2*n1**2)))/(112.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_7(n1, f):
            return (-9*f**2*(5 + 64*n1 - 12*n1**2 + 4*n1**3))/(224.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_8(n1, f):
            return (-405*f**2)/(224.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_9(n1, f):
            return (9*f**2*(5 + 24*n1 + 4*n1**2))/(112.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_10(n1, f):
            return (-9*f**2*(5 - 16*n1 + 24*n1**2))/(224.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_11(n1, f):
            return (-9*f)/(56.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1)

        def M13direct_12(n1, f):
            return (27*f)/(56.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1)

        def M13direct_13(n1, f):
            return (25*(-1 + 2*n1))/(56.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_14(n1, f):
            return (-15*(-5 - 6*n1 + 4*n1**2))/(56.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_15(n1, f):
            return (15*(-20 + 45*f + 84*n1 - 16*n1**2))/(448.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_16(n1, f):
            return (-45*(-10 + 42*n1 - 8*n1**2 + 5*f*(5 + 2*n1)))/(224.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_17(n1, f):
            return (375*f*(1 + 4*n1))/(448.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_18(n1, f):
            return (39*(-1 + 4*n1))/(224.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_19(n1, f):
            return (2031 + 1168*n1 - 2988*n1**2 + 560*n1**3)/(672.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_20(n1, f):
            return 15/(56.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1)

        def M13direct_21(n1, f):
            return -45/(56.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1)

        def M13direct_22(n1, f):
            return (25*(-1 + 2*n1))/(112.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_23(n1, f):
            return (-15*(-5 - 6*n1 + 4*n1**2))/(112.*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_24(n1, f):
            return (675*f)/(448.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_25(n1, f):
            return (-15*f*(5 + 24*n1 + 4*n1**2))/(224.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M13direct_26(n1, f):
            return (15*f*(5 - 16*n1 + 24*n1**2))/(448.*(-5 + n1)*(-4 + n1)*(-3 + n1)*(-2 + n1)*(-1 + n1)*n1*(1 + n1))

        def M22direct_0(n1, n2, f):
            return ((147*f**2*(18 - 12*n1**3 + 4*n1**4 + 21*n2 - 13*n2**2 - 12*n2**3 + 4*n2**4 + n1*(21 + 2*n2 - 4*n2**2) + n1**2*(-13 - 4*n2 + 8*n2**2)))/((-4 + n1 + n2)*(-3 + n1 + n2)) - (28*f*(-18 + 28*n1**4 + 135*n2 - 55*n2**2 - 60*n2**3 + 28*n2**4 + n1**3*(-60 + 56*n2) + n1**2*(-55 - 88*n2 + 56*n2**2) + n1*(135 - 122*n2 - 88*n2**2 + 56*n2**3)))/(-3 + n1 + n2) + (8*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(58 + 98*n1**3*n2 + (3 - 91*n2)*n2 + 7*n1**2*(-13 - 2*n2 + 28*n2**2) + n1*(3 + 2*n2*(-73 + 7*n2*(-1 + 7*n2)))))/((-1 + 2*n1)*(-1 + 2*n2)))/(1568.*n1*(1 + n1)*n2*(1 + n2))

        def M22direct_1(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(7*f*(-6 + 8*n1**6 + 41*n2 - 61*n2**2 - 10*n2**3 + 50*n2**4 - 40*n2**5 + 8*n2**6 - 8*n1**5*(5 + 4*n2) + 2*n1**4*(25 + 52*n2 + 12*n2**2) - 2*n1**3*(5 - 14*n2 + 32*n2**3) + n1**2*(-61 - 106*n2 - 44*n2**2 + 24*n2**4) + n1*(41 + 58*n2 - 106*n2**2 + 28*n2**3 + 104*n2**4 - 32*n2**5)) + 2*(-168 + 662*n2 + 112*n1**6*n2 - 391*n2**2 - 821*n2**3 + 652*n2**4 - 108*n2**5 + 4*n1**5*(-27 - 184*n2 + 84*n2**2) + 4*n1**4*(163 + 222*n2 - 396*n2**2 + 112*n2**3) + n1**3*(-821 + 1448*n2 + 1012*n2**2 - 1696*n2**3 + 448*n2**4) + n1**2*(-391 - 2491*n2 + 2936*n2**2 + 1012*n2**3 - 1584*n2**4 + 336*n2**5) + n1*(662 + 226*n2 - 2491*n2**2 + 1448*n2**3 + 888*n2**4 - 736*n2**5 + 112*n2**6))))/(112.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2)*(-1 + 2*n2))

        def M22direct_2(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(18 - 12*n1**3 + 4*n1**4 + 21*n2 - 13*n2**2 - 12*n2**3 + 4*n2**4 + n1*(21 + 2*n2 - 4*n2**2) + n1**2*(-13 - 4*n2 + 8*n2**2)))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_3(n1, n2, f):
            return (28*n1**3 - 2*n1**2*(71 + 7*f - 42*n2) + (-3 + 2*n2)*(24 - (50 + 7*f)*n2 + 14*n2**2) + n1*(198 + 21*f - 284*n2 + 84*n2**2))/(28.*n1*n2*(-3 + n1 + n2))

        def M22direct_4(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(-3*n1 + 2*n1**2 + n2*(-3 + 2*n2)))/(4.*n1*n2*(-3 + n1 + n2))

        def M22direct_5(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(-18 - 14*n1**3 + n1**2*(37 - 42*n2) + 21*n2 + 37*n2**2 - 14*n2**3 + n1*(21 + 74*n2 - 42*n2**2) + 7*f*(-6 - n1 + 2*n1**2 - n2 + 2*n2**2)))/(14.*n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_6(n1, n2, f):
            return -0.5*(f*(-3 + 2*n1 + 2*n2)*(2*n1**3 + n1**2*(-1 + 2*n2) + 2*n1*(-3 - n2 + n2**2) + n2*(-6 - n2 + 2*n2**2)))/(n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_7(n1, n2, f):
            return (3*f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-35*f*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2) + 2*(30 + 14*n1**3 + 79*n2 - 87*n2**2 + 14*n2**3 + n1**2*(-87 + 42*n2) + n1*(79 - 174*n2 + 42*n2**2))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_8(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(28*f*(1014 - 748*n1 + 56*n1**2 + 4*(50 - 21*n1)*n2 + 56*n2**2 + (450*(-5 + 2*n1)*(-3 + 2*n1))/(-4 + n1 + n2) - (213*(3 - 2*n1)**2)/(-3 + n1 + n2)) - (16*(-1 + 2*n1 + 2*n2)*(46 + 98*n1**3*n2 + (13 - 63*n2)*n2 + 7*n1**2*(-9 + 2*n2*(-5 + 14*n2)) + n1*(13 + 2*n2*(-69 + 7*n2*(-5 + 7*n2)))))/((-1 + 2*n1)*(-1 + 2*n2)) - (147*f**2*(-330 + 28*n1**4 - 2*n1**3*(43 + 32*n2) + n1**2*(62 + 66*n2 + 56*n2**2) + n1*(-19 + 2*n2*(72 + (33 - 32*n2)*n2)) + n2*(-19 + 2*n2*(31 + n2*(-43 + 14*n2)))))/((-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))))/(1568.*n1*(1 + n1)*n2*(1 + n2))

        def M22direct_9(n1, n2, f):
            return -0.004464285714285714*(f**2*(-3 + 2*n1 + 2*n2)*(7*f*(390 + 32*n1**7 - 823*n2 + 384*n2**2 - 184*n2**3 - 1232*n2**4 + 936*n2**5 - 256*n2**6 + 32*n2**7 - 32*n1**6*(8 + 7*n2) + 8*n1**5*(117 + 116*n2 + 20*n2**2) - 8*n1**4*(154 + 219*n2 - 80*n2**2 + 12*n2**3) - 8*n1**3*(23 - 66*n2 + 240*n2**2 + 16*n2**3 + 12*n2**4) + 8*n1**2*(48 + 521*n2 + 80*n2**2 - 240*n2**3 + 80*n2**4 + 20*n2**5) + n1*(-823 - 2292*n2 + 4168*n2**2 + 528*n2**3 - 1752*n2**4 + 928*n2**5 - 224*n2**6)) + 2*(448*n1**7*n2 + 32*n1**6*(-15 - 159*n2 + 56*n2**2) + 16*n1**5*(294 + 1149*n2 - 1052*n2**2 + 196*n2**3) + 4*n1**4*(-3741 - 4174*n2 + 12500*n2**2 - 6264*n2**3 + 896*n2**4) + 4*n1**3*(4359 - 9082*n2 - 8542*n2**2 + 15568*n2**3 - 6264*n2**4 + 784*n2**5) + 3*(1530 - 4191*n2 + 417*n2**2 + 5812*n2**3 - 4988*n2**4 + 1568*n2**5 - 160*n2**6) + n1**2*(1251 + 64612*n2 - 59528*n2**2 - 34168*n2**3 + 50000*n2**4 - 16832*n2**5 + 1792*n2**6) + n1*(-12573 - 10218*n2 + 64612*n2**2 - 36328*n2**3 - 16696*n2**4 + 18384*n2**5 - 5088*n2**6 + 448*n2**7))))/(n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2)*(-1 + 2*n2))

        def M22direct_10(n1, n2, f):
            return -0.03125*(f**3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(-150 - 26*n1**3 + 4*n1**4 + 35*n2 + 50*n2**2 - 26*n2**3 + 4*n2**4 + n1*(35 + 16*n2 - 18*n2**2) + 2*n1**2*(25 - 9*n2 + 4*n2**2)))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_11(n1, n2, f):
            return (3*f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-35*f*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2) + 2*(30 + 14*n1**3 + 79*n2 - 87*n2**2 + 14*n2**3 + n1**2*(-87 + 42*n2) + n1*(79 - 174*n2 + 42*n2**2))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_12(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-63*f*(-30 + 8*n1**4 - 8*n1**3*(2 + 3*n2) + 4*n1**2*(-2 + n2*(9 + 4*n2)) + n2*(-29 + 8*n2*(-1 + (-2 + n2)*n2)) + n1*(-29 + 4*n2*(11 + 9*n2 - 6*n2**2))) + 4*(-5 + n1 + n2)*(-126 + 28*n1**4 - 2*n1**3*(9 + 28*n2) + n1**2*(92 + 6*(37 - 28*n2)*n2) + n2*(-153 + 2*n2*(46 + n2*(-9 + 14*n2))) - n1*(153 + 2*n2*(16 + n2*(-111 + 28*n2))))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_13(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(21*f*(-30 + 8*n1**4 - 29*n2 - 8*n2**2 - 16*n2**3 + 8*n2**4 - 8*n1**3*(2 + 3*n2) + 4*n1**2*(-2 + 9*n2 + 4*n2**2) + n1*(-29 + 44*n2 + 36*n2**2 - 24*n2**3)) + 2*(-1170 - 981*n2 + 763*n2**2 + 36*n2**3 - 28*n2**4 + 28*n1**4*(-1 + 2*n2) + 4*n1**3*(9 - 127*n2 + 42*n2**2) + n1**2*(763 + 856*n2 - 960*n2**2 + 168*n2**3) + n1*(-981 + 686*n2 + 856*n2**2 - 508*n2**3 + 56*n2**4))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_14(n1, n2, f):
            return (f**3*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_15(n1, n2, f):
            return (f*(21*f*(18 - 12*n1**3 + 4*n1**4 + 21*n2 - 13*n2**2 - 12*n2**3 + 4*n2**4 + n1*(21 + 2*n2 - 4*n2**2) + n1**2*(-13 - 4*n2 + 8*n2**2)) - 2*(72 + 28*n1**5 - 558*n2 + 355*n2**2 + 185*n2**3 - 172*n2**4 + 28*n2**5 + 4*n1**4*(-43 + 21*n2) + n1**3*(185 - 372*n2 + 112*n2**2) + n1**2*(355 + 175*n2 - 400*n2**2 + 112*n2**3) + n1*(-558 + 758*n2 + 175*n2**2 - 372*n2**3 + 84*n2**4))))/(112.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_16(n1, n2, f):
            return -0.017857142857142856*(f*(-3 + 2*n1 + 2*n2)*(21*f*(2*n1**4 - n1**3*(5 + 4*n2) + (-3 + n2)*(1 + n2)*(2 + n2*(-1 + 2*n2)) - n1*(1 + n2)*(1 + n2*(-9 + 4*n2)) + n1**2*(-2 + n2*(5 + 4*n2))) + (-4 + n1 + n2)*(78 + 14*n1**3*(1 + 4*n2) + n2*(19 + n2*(-117 + 14*n2)) + n1**2*(-117 + 2*n2*(-37 + 56*n2)) + n1*(19 + 2*n2*(-75 + n2*(-37 + 28*n2))))))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_17(n1, n2, f):
            return -0.0625*(f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(30 + 23*n2 - 9*n2**2 - 2*n2**3 + n1**3*(-2 + 8*n2) - n1**2*(9 + 14*n2) + n1*(23 - 14*n2 - 14*n2**2 + 8*n2**3)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_18(n1, n2, f):
            return (-7*f*(2106 - 1173*n2 - 730*n2**2 + 468*n2**3 - 56*n2**4 + 8*n1**4*(-7 + 2*n2) + 4*n1**3*(117 - 68*n2 + 4*n2**2) + 2*n1**2*(-365 + 348*n2 - 80*n2**2 + 8*n2**3) + n1*(-1173 + 216*n2 + 696*n2**2 - 272*n2**3 + 16*n2**4)) + 4*(-288 + 36*n2 + 1133*n2**2 - 1097*n2**3 + 316*n2**4 - 28*n2**5 + 28*n1**5*(-1 + 2*n2) + 4*n1**4*(79 - 183*n2 + 56*n2**2) + n1**3*(-1097 + 3270*n2 - 2056*n2**2 + 336*n2**3) + n1**2*(1133 - 5589*n2 + 5908*n2**2 - 2056*n2**3 + 224*n2**4) + n1*(36 + 2914*n2 - 5589*n2**2 + 3270*n2**3 - 732*n2**4 + 56*n2**5)))/(1008.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_19(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(7*f*(-54 - 177*n2 + 4*n2**2 + 8*n2**4 + 8*n1**4*(1 + n2) + 4*n1**3*n2*(-25 + 2*n2) + 4*n1**2*(1 + 33*n2 - 2*n2**2 + 2*n2**3) + n1*(-177 + 180*n2 + 132*n2**2 - 100*n2**3 + 8*n2**4)) + 6*(-48 - 152*n2 + 53*n2**2 + 53*n2**3 - 14*n2**4 + 14*n1**4*(-1 + 2*n2) + n1**3*(53 - 226*n2 + 84*n2**2) + n1**2*(53 + 433*n2 - 424*n2**2 + 84*n2**3) + n1*(-152 - 62*n2 + 433*n2**2 - 226*n2**3 + 28*n2**4))))/(504.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_20(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(42 + 16*n1**4*(-1 + n2) - 113*n2 + 40*n2**2 + 44*n2**3 - 16*n2**4 + 4*n1**3*(11 - 16*n2 + 4*n2**2) + 8*n1**2*(5 + 5*n2 - 10*n2**2 + 2*n2**3) + n1*(-113 + 124*n2 + 40*n2**2 - 64*n2**3 + 16*n2**4)))/(48.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_21(n1, n2, f):
            return (21*f*(-3 + 2*n1)*(-3 + 2*n2)*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2) - 8*(36 + 14*n1**4*(-1 + n2) + 27*n2 - 133*n2**2 + 87*n2**3 - 14*n2**4 + 3*n1**3*(29 - 45*n2 + 14*n2**2) + n1**2*(-133 + 356*n2 - 242*n2**2 + 42*n2**3) + n1*(27 - 278*n2 + 356*n2**2 - 135*n2**3 + 14*n2**4)))/(112.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_22(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(7*f*(-606 + 28*n1**4 - 149*n2 + 152*n2**2 - 82*n2**3 + 28*n2**4 - 2*n1**3*(41 + 56*n2) + 2*n1**2*(76 + 71*n2 + 28*n2**2) + n1*(-149 + 236*n2 + 142*n2**2 - 112*n2**3)) + 4*(14*n1**4*(5 + 7*n2) + n1**3*(-605 - 315*n2 + 294*n2**2) + n1**2*(1265 - 856*n2 - 770*n2**2 + 294*n2**3) + 5*(-156 + 67*n2 + 253*n2**2 - 121*n2**3 + 14*n2**4) + n1*(335 + 1942*n2 - 856*n2**2 - 315*n2**3 + 98*n2**4))))/(392.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_23(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(-42 + 869*n2 + 270*n2**2 - 332*n2**3 + 24*n2**4 + 8*n1**4*(3 + 14*n2) + 4*n1**3*(-83 - 44*n2 + 28*n2**2) + 2*n1**2*(135 - 248*n2 - 144*n2**2 + 56*n2**3) + n1*(869 + 624*n2 - 496*n2**2 - 176*n2**3 + 112*n2**4)))/(112.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_24(n1, n2, f):
            return 0.5

        def M22direct_25(n1, n2, f):
            return (3 - 2*n1 - 2*n2)/(n1*n2)

        def M22direct_26(n1, n2, f):
            return (3*f**2*(-3 + 2*n1)*(-3 + 2*n2))/(8.*n1*n2*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_27(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(-2*(-4 + n1 + n2)*(-3 + n1 + n2)*(-8 + 7*n1 + 7*n2) + 21*f*(3 + n1**2 + (-1 + n2)*n2 - n1*(1 + 2*n2))))/(28.*n1*n2*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_28(n1, n2, f):
            return -0.125*(f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(15 + 2*(-4 + n1)*n1 + 2*(-4 + n2)*n2))/(n1*n2*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_29(n1, n2, f):
            return (3*f**2*(-3 + 2*n1)*(-3 + 2*n2))/(8.*n1*n2*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_30(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-3 + 2*n1**2 + n1*(4 - 8*n2) + 2*n2*(2 + n2)))/(4.*n1*n2*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_31(n1, n2, f):
            return (f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(8.*n1*n2*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_32(n1, n2, f):
            return (f*((3 - 2*n1)*n1 + (3 - 2*n2)*n2))/(4.*n1*n2*(-3 + n1 + n2))

        def M22direct_33(n1, n2, f):
            return -0.25*(f*(-3 + 2*n1 + 2*n2)*(-3*n2 + n1*(-3 + 4*n2)))/(n1*n2*(-3 + n1 + n2))

        def M22direct_34(n1, n2, f):
            return ((-3 + 2*n1)*(-3 + 2*n2)*(-9 + 2*n1 + 2*n2))/(36.*n1*n2*(-3 + n1 + n2))

        def M22direct_35(n1, n2, f):
            return ((-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2))/(12.*n1*n2*(-3 + n1 + n2))

        def M22direct_36(n1, n2, f):
            return (-9 + n1*(6 - 4*n2) + 6*n2)/(4.*n1*n2*(-3 + n1 + n2))

        def M22direct_37(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(-39 - 8*n2 + 4*n1*(-2 + 7*n2)))/(28.*n1*n2*(-3 + n1 + n2))

        def M22direct_38(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(n1*(1 + n1)*n2*(1 + n2))

        def M22direct_39(n1, n2, f):
            return (-9*f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_40(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(2*(-4 + n1 + n2)*(-3 + n1 + n2)*(-1 + 2*n1 + 2*n2)*(-2 + 7*n1 + 7*n2) - 21*f*(-15 + 4*n1**3 - 2*n1**2*(3 + 2*n2) - 4*n1*(-5 + n2*(2 + n2)) + 2*n2*(10 + n2*(-3 + 2*n2)))))/(28.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_41(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(33 + 16*n2 + 4*(n1*(4 + (-2 + n1)*n1) + (-11 + n1)*n1*n2 + (-2 + n1)*n2**2 + n2**3)))/(8.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_42(n1, n2, f):
            return (-9*f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_43(n1, n2, f):
            return -0.25*(f**2*(-3 + 2*n1 + 2*n2)*(-21 + 8*n1**3 + n1**2*(12 - 24*n2) + n1*(-2 + 12*(3 - 2*n2)*n2) + 2*n2*(-1 + 6*n2 + 4*n2**2)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_44(n1, n2, f):
            return -0.125*(f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(15 + 8*n1**2*(-2 + n2) + 4*(1 - 4*n2)*n2 + n1*(4 - 4*n2 + 8*n2**2)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_45(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(n1*(-1 + 2*n1) + (-2 + n2)*(3 + 2*n2)))/(2.*n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_46(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(6 - n2*(5 + 2*n2) + n1**2*(-2 + 4*n2) + n1*(-5 + 4*(-1 + n2)*n2)))/(2.*n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_47(n1, n2, f):
            return -0.05555555555555555*((-3 + 2*n1 + 2*n2)*(-57 + 4*n1**2*(-2 + n2) + 52*n2 - 8*n2**2 + 4*n1*(13 + (-10 + n2)*n2)))/(n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_48(n1, n2, f):
            return -0.16666666666666666*((-3 + 2*n1 + 2*n2)*(-3 + 4*n1**2*(-2 + n2) + 4*n1*(-2 + n2)**2 - 8*(-2 + n2)*n2))/(n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_49(n1, n2, f):
            return ((-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2))/(2.*n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_50(n1, n2, f):
            return -0.07142857142857142*((-3 + 2*n1 + 2*n2)*(81 + 6*n1*(-21 + 2*n1) - 126*n2 + 4*n1*(6 + 7*n1)*n2 + 4*(3 + 7*n1)*n2**2))/(n1*(1 + n1)*n2*(1 + n2)*(-3 + n1 + n2))

        def M22direct_51(n1, n2, f):
            return (105*f**4*(-5 + 2*n1)*(-3 + 2*n1)*(-5 + 2*n2)*(-3 + 2*n2))/(128.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_52(n1, n2, f):
            return (3*f**3*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(245*f*(5 + n1**2 - n2 + n2**2 - n1*(1 + 2*n2)) - 2*(-300 + 7*n1**3 + 320*n2 - 87*n2**2 + 7*n2**3 + 3*n1**2*(-29 + 7*n2) + n1*(320 - 174*n2 + 21*n2**2))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_53(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*((-20580*f**2*(-3 + 2*n1)*(-1 + 2*n1 + 2*n2))/(-3 + n1 + n2) + (30870*f**2*(-5 + 2*n1)*(-3 + 2*n1)*(-1 + 2*n1 + 2*n2))/((-4 + n1 + n2)*(-3 + n1 + n2)) - (20580*f**2*(-7 + 2*n1)*(-5 + 2*n1)*(-3 + 2*n1)*(-1 + 2*n1 + 2*n2))/((-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2)) + (5145*f**2*(-9 + 2*n1)*(-7 + 2*n1)*(-5 + 2*n1)*(-3 + 2*n1)*(-1 + 2*n1 + 2*n2))/((-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2)) - 168*f*(-1 + 2*n1 + 2*n2)*(-2 + 7*n1 + 7*n2) + (16*(-1 + 2*n1 + 2*n2)*(50 - n1*(9 + 35*n1) - 9*n2 + 2*n1*(-33 + 7*n1*(-9 + 7*n1))*n2 + 7*(-5 + 2*n1*(-9 + 14*n1))*n2**2 + 98*n1*n2**3))/((-1 + 2*n1)*(-1 + 2*n2)) + (672*f*(-1 + 2*n1 + 2*n2)*(3 - 7*n2 + n1*(-16 + 7*n1 + 7*n2)))/(-3 + n1 + n2) + 42*f*(119*f*(-1 + 2*n1 + 2*n2) - (4*(-3 + 2*n1)*(-6 + n1*(101 + 4*n1*(-30 + 7*n1)) + 57*n2 + 2*n1*(-81 + 28*n1)*n2 + 14*(-3 + 2*n1)*n2**2))/((-4 + n1 + n2)*(-3 + n1 + n2)))))/(3136.*n1*(1 + n1)*n2*(1 + n2))

        def M22direct_54(n1, n2, f):
            return (f**3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(7*f*(495 + 8*n1**7 + 681*n2 + 937*n2**2 + 2342*n2**3 - 1250*n2**4 + 524*n2**5 - 92*n2**6 + 8*n2**7 - 4*n1**6*(23 + 22*n2) + 4*n1**5*(131 + 156*n2 + 30*n2**2) - 10*n1**4*(125 + 258*n2 - 30*n2**2 + 4*n2**3) + n1**3*(2342 + 680*n2 + 280*n2**2 - 1280*n2**3 - 40*n2**4) + n1**2*(937 - 4622*n2 + 10020*n2**2 + 280*n2**3 + 300*n2**4 + 120*n2**5) + n1*(681 - 6098*n2 - 4622*n2**2 + 680*n2**3 - 2580*n2**4 + 624*n2**5 - 88*n2**6)) + 2*(112*n1**7*n2 + 4*n1**6*(-33 - 466*n2 + 112*n2**2) + 2*n1**5*(975 + 5580*n2 - 3132*n2**2 + 392*n2**3) + n1**4*(-9930 - 28650*n2 + 32372*n2**2 - 9472*n2**3 + 896*n2**4) + n1**3*(20325 + 23936*n2 - 75476*n2**2 + 42160*n2**3 - 9472*n2**4 + 784*n2**5) - 3*(-4500 + 1920*n2 + 4671*n2**2 - 6775*n2**3 + 3310*n2**4 - 650*n2**5 + 44*n2**6) + n1**2*(-14013 + 29359*n2 + 61012*n2**2 - 75476*n2**3 + 32372*n2**4 - 6264*n2**5 + 448*n2**6) + n1*(-5760 - 56346*n2 + 29359*n2**2 + 23936*n2**3 - 28650*n2**4 + 11160*n2**5 - 1864*n2**6 + 112*n2**7))))/(224.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2)*(-1 + 2*n2))

        def M22direct_55(n1, n2, f):
            return (f**4*(5265 + 64*n1**8 + 256*n1**7*(-3 + n2) - 4428*n2 - 21276*n2**2 + 19248*n2**3 - 44*n2**4 - 5952*n2**5 + 3616*n2**6 - 768*n2**7 + 64*n2**8 + 32*n1**6*(113 - 112*n2 + 16*n2**2) + 64*n1**5*(-93 + 263*n2 - 112*n2**2 + 12*n2**3) + 4*n1**4*(-11 - 8560*n2 + 8536*n2**2 - 2240*n2**3 + 224*n2**4) + 16*n1**3*(1203 + 840*n2 - 4560*n2**2 + 2616*n2**3 - 560*n2**4 + 48*n2**5) + 4*n1**2*(-5319 + 14688*n2 + 6778*n2**2 - 18240*n2**3 + 8536*n2**4 - 1792*n2**5 + 128*n2**6) + 4*n1*(-1107 - 11412*n2 + 14688*n2**2 + 3360*n2**3 - 8560*n2**4 + 4208*n2**5 - 896*n2**6 + 64*n2**7)))/(128.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_56(n1, n2, f):
            return (105*f**4*(-5 + 2*n1)*(-3 + 2*n1)*(-5 + 2*n2)*(-3 + 2*n2))/(64.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_57(n1, n2, f):
            return (3*f**3*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-2*(-6 + n1 + n2)*(-5 + n1 + n2)*(-10 + 7*n1 + 7*n2) + 35*f*(40 + 13*n1**2 - 3*n1*(1 + 10*n2) + n2*(-3 + 13*n2))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_58(n1, n2, f):
            return (f**3*(-3 + 2*n1 + 2*n2)*(105*f*(-45 + 8*n1**5 + 69*n2 - 29*n2**2 + 148*n2**3 - 16*n2**4 + 8*n2**5 - 16*n1**4*(1 + 3*n2) + 4*n1**3*(37 + 13*n2 + 10*n2**2) + n1**2*(-29 - 328*n2 + 24*n2**2 + 40*n2**3) + n1*(69 + 310*n2 - 328*n2**2 + 52*n2**3 - 48*n2**4)) - 4*(-1260 - 358*n1**5 + 28*n1**6 - 3888*n2 + 2873*n2**2 - 2129*n2**3 + 1434*n2**4 - 358*n2**5 + 28*n2**6 + n1**4*(1434 + 526*n2 - 252*n2**2) - n1**3*(2129 + 4716*n2 - 3368*n2**2 + 448*n2**3) + n1**2*(2873 + 10557*n2 - 12300*n2**2 + 3368*n2**3 - 252*n2**4) + n1*(-3888 - 2174*n2 + 10557*n2**2 - 4716*n2**3 + 526*n2**4))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_59(n1, n2, f):
            return -0.004464285714285714*(f**3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(7*f*(24*n1**5 - 96*n1**4*(1 + n2) + n1**3*(432 + 92*n2 + 40*n2**2) + n1**2*(381 - 1084*n2 + 600*n2**2 + 40*n2**3) + n1*(429 - 574*n2 - 1084*n2**2 + 92*n2**3 - 96*n2**4) + 3*(60 + 143*n2 + 127*n2**2 + 144*n2**3 - 32*n2**4 + 8*n2**5)) + 2*(5220 + 4896*n2 - 3523*n2**2 - 173*n2**3 + 268*n2**4 - 28*n2**5 + 28*n1**5*(-1 + 2*n2) + 4*n1**4*(67 - 226*n2 + 56*n2**2) + n1**3*(-173 + 4436*n2 - 2572*n2**2 + 336*n2**3) + n1**2*(-3523 - 5575*n2 + 8336*n2**2 - 2572*n2**3 + 224*n2**4) + n1*(4896 - 5366*n2 - 5575*n2**2 + 4436*n2**3 - 904*n2**4 + 56*n2**5))))/(n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_60(n1, n2, f):
            return -0.015625*(f**4*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(3 + 2*n1 + 2*n2)*(35 - 12*n1 + 2*n1**2 - 12*n2 + 2*n2**2))/(n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_61(n1, n2, f):
            return (-15*f**3*(-3 + 2*n1)*(-3 + 2*n2)*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_62(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(4*(-5 + n1 + n2)*(-4 + n1 + n2)*(-9 + 7*n1 + 7*n2)*(n1*(-1 + 2*n1) + (-2 + n2)*(3 + 2*n2)) - 21*f*(12*n1**4 + n1**3*(6 - 56*n2) + 2*n1**2*(-6 + n2*(-33 + 52*n2)) + 3*(-2 + n2)*(45 + 2*n2*(8 + n2*(5 + 2*n2))) + n1*(39 - 2*n2*(-118 + n2*(33 + 28*n2))))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_63(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(7*f*(-30 + 24*n1**5 + 31*n2 - 170*n2**2 + 500*n2**3 - 100*n2**4 + 24*n2**5 - 4*n1**4*(25 + 22*n2) + 4*n1**3*(125 + 4*n2 + 24*n2**2) + 2*n1**2*(-85 - 268*n2 - 4*n2**2 + 48*n2**3) + n1*(31 + 600*n2 - 536*n2**2 + 16*n2**3 - 88*n2**4)) + 4*(600 + 910*n2 - 2521*n2**2 + 1248*n2**3 - 227*n2**4 + 14*n2**5 + 14*n1**5*(1 + 4*n2) + n1**4*(-227 - 582*n2 + 224*n2**2) + 4*n1**3*(312 + 407*n2 - 454*n2**2 + 84*n2**3) + n1**2*(-2521 + 28*n2 + 3710*n2**2 - 1816*n2**3 + 224*n2**4) + 2*n1*(455 - 1681*n2 + 14*n2**2 + 814*n2**3 - 291*n2**4 + 28*n2**5))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_64(n1, n2, f):
            return (f**3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(-150 + 35*n2 - 12*n2**2 - 2*n2**3 - 6*n1**2*(2 + 7*n2) + n1**3*(-2 + 8*n2) + n1*(35 + 140*n2 - 42*n2**2 + 8*n2**3)))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_65(n1, n2, f):
            return (f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-865 + 508*n2 - 52*n2**2 + n1**2*(-52 + 8*n2) + 4*n1*(127 - 51*n2 + 2*n2**2)))/(96.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_66(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(21*f*(-3075 + 3110*n2 - 1354*n2**2 + 640*n2**3 - 56*n2**4 + 8*n1**4*(-7 + 2*n2) - 8*n1**3*(-80 + 33*n2 + 2*n2**2) - 2*n1**2*(677 + 108*n2 - 312*n2**2 + 8*n2**3) + 2*n1*(1555 - 654*n2 - 108*n2**2 - 132*n2**3 + 8*n2**4)) - 8*(4080 - 6376*n2 + 4507*n2**2 - 1524*n2**3 + 239*n2**4 - 14*n2**5 + 14*n1**5*(-1 + 2*n2) + n1**4*(239 - 480*n2 + 112*n2**2) + 2*n1**3*(-762 + 1544*n2 - 685*n2**2 + 84*n2**3) + n1**2*(4507 - 9082*n2 + 5698*n2**2 - 1370*n2**3 + 112*n2**4) + 2*n1*(-3188 + 6007*n2 - 4541*n2**2 + 1544*n2**3 - 240*n2**4 + 14*n2**5))))/(2016.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_67(n1, n2, f):
            return -0.000496031746031746*(f*(-3 + 2*n1 + 2*n2)*(7*f*(-975 + 1600*n2 + 224*n2**2 + 960*n2**3 - 32*n2**4 + 32*n2**5 + 32*n1**5*(1 + n2) + 16*n1**4*(-2 - 43*n2 + 4*n2**2) + 16*n1**3*(60 + 134*n2 - 27*n2**2 + 4*n2**3) + 16*n1**2*(14 - 443*n2 + 217*n2**2 - 27*n2**3 + 4*n2**4) + 8*n1*(200 + 621*n2 - 886*n2**2 + 268*n2**3 - 86*n2**4 + 4*n2**5)) + 24*(480 - 676*n2 + 811*n2**2 - 564*n2**3 + 155*n2**4 - 14*n2**5 + 14*n1**5*(-1 + 2*n2) + n1**4*(155 - 396*n2 + 112*n2**2) + 2*n1**3*(-282 + 944*n2 - 559*n2**2 + 84*n2**3) + n1**2*(811 - 3550*n2 + 3466*n2**2 - 1118*n2**3 + 112*n2**4) + 2*n1*(-338 + 1231*n2 - 1775*n2**2 + 944*n2**3 - 198*n2**4 + 14*n2**5))))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_68(n1, n2, f):
            return -0.010416666666666666*(f**2*(225 + 64*n1**6*(-1 + n2) - 540*n2 - 514*n2**2 + 2052*n2**3 - 1528*n2**4 + 432*n2**5 - 64*n2**6 + 48*n1**5*(9 - 14*n2 + 4*n2**2) + 8*n1**4*(-191 + 408*n2 - 240*n2**2 + 32*n2**3) + 4*n1**3*(513 - 1822*n2 + 1916*n2**2 - 656*n2**3 + 64*n2**4) + 2*n1**2*(-257 + 2938*n2 - 5784*n2**2 + 3832*n2**3 - 960*n2**4 + 96*n2**5) + 4*n1*(-135 - 179*n2 + 1469*n2**2 - 1822*n2**3 + 816*n2**4 - 168*n2**5 + 16*n2**6)))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_69(n1, n2, f):
            return (-15*f**2*(-5 + 2*n1)*(-3 + 2*n1)*(-5 + 2*n2)*(-3 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_70(n1, n2, f):
            return (f*(-3 + 2*n1 + 2*n2)*(-3*f*(-3 + 2*n1)*(-3 + 2*n2)*(685 + 98*n1**2 - 102*n2 + 98*n2**2 - 2*n1*(51 + 112*n2)) + 16*(-300 + 7*n1**4*(-1 + n2) + 475*n2 - 308*n2**2 + 80*n2**3 - 7*n2**4 + n1**3*(80 - 100*n2 + 21*n2**2) + n1**2*(-308 + 461*n2 - 186*n2**2 + 21*n2**3) + n1*(475 - 796*n2 + 461*n2**2 - 100*n2**3 + 7*n2**4))))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_71(n1, n2, f):
            return -0.0006377551020408163*(f*(-3 + 2*n1 + 2*n2)*(7*f*(-10365 + 112*n1**5 + 15998*n2 - 6140*n2**2 + 5640*n2**3 - 760*n2**4 + 112*n2**5 - 8*n1**4*(95 + 98*n2) + 8*n1**3*(705 + 196*n2 + 56*n2**2) + 4*n1**2*(-1535 - 2322*n2 + 744*n2**2 + 112*n2**3) - 2*n1*(-7999 + 1160*n2 + 4644*n2**2 - 784*n2**3 + 392*n2**4)) + 16*(660 + 4043*n2 - 8900*n2**2 + 4758*n2**3 - 979*n2**4 + 70*n2**5 + 14*n1**5*(5 + 7*n2) + n1**4*(-979 - 791*n2 + 392*n2**2) + n1**3*(4758 + 522*n2 - 2723*n2**2 + 588*n2**3) + n1**2*(-8900 + 7771*n2 + 3002*n2**2 - 2723*n2**3 + 392*n2**4) + n1*(4043 - 14860*n2 + 7771*n2**2 + 522*n2**3 - 791*n2**4 + 98*n2**5))))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_72(n1, n2, f):
            return -0.004464285714285714*(f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(8*n1**4*(3 + 14*n2) + 8*n1**3*(-63 - 57*n2 + 14*n2**2) + 2*n1**2*(441 + 260*n2 - 536*n2**2 + 56*n2**3) + 3*(-1275 - 710*n2 + 294*n2**2 - 168*n2**3 + 8*n2**4) + 2*n1*(-1065 + 2786*n2 + 260*n2**2 - 228*n2**3 + 56*n2**4)))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_73(n1, n2, f):
            return (105*f**4*(-5 + 2*n1)*(-3 + 2*n1)*(-5 + 2*n2)*(-3 + 2*n2))/(128.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_74(n1, n2, f):
            return (15*f**4*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(5 + 6*n1**2 + n1*(4 - 16*n2) + 4*n2 + 6*n2**2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_75(n1, n2, f):
            return (3*f**4*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-45 + 8*n1**4 + 24*n2 + 52*n2**2 + 96*n2**3 + 8*n2**4 - 32*n1**3*(-3 + 4*n2) + 4*n1**2*(13 - 48*n2 + 72*n2**2) - 8*n1*(-3 - 8*n2 + 24*n2**2 + 16*n2**3)))/(64.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_76(n1, n2, f):
            return (f**4*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(5 + 6*n1**2 + n1*(4 - 16*n2) + 4*n2 + 6*n2**2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_77(n1, n2, f):
            return (f**4*(-5 + 2*n1)*(-3 + 2*n1)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(3 + 2*n1 + 2*n2)*(15 - 16*n2 + 4*n2**2))/(128.*n1*(1 + n1)*n2*(1 + n2)*(-6 + n1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_78(n1, n2, f):
            return (-15*f**3*(-3 + 2*n1)*(-3 + 2*n2)*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_79(n1, n2, f):
            return (-3*f**3*(-3 + 2*n1 + 2*n2)*(8*n1**4 + n1**3*(44 - 64*n2) + (-2 + n2)*(15 + 22*n2 + 60*n2**2 + 8*n2**3) + n1**2*(-98 + 24*n2*(-1 + 4*n2)) - n1*(29 + 8*n2*(-28 + n2*(3 + 8*n2)))))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_80(n1, n2, f):
            return -0.03125*(f**3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-30 - 29*n2 + 12*n2**2 - 64*n2**3 + 8*n1**3*(-8 + 7*n2) + n1**2*(12 + 84*n2 - 128*n2**2) + n1*(-29 + 4*n2 + 84*n2**2 + 56*n2**3)))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_81(n1, n2, f):
            return -0.03125*(f**3*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(-10 - 3*n2 + n1*(-3 + 4*n2)))/(n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_82(n1, n2, f):
            return (f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-865 + 508*n2 - 52*n2**2 + n1**2*(-52 + 8*n2) + 4*n1*(127 - 51*n2 + 2*n2**2)))/(96.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_83(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-2865 - 92*n2 + 328*n2**2 + 1088*n2**3 - 112*n2**4 + 16*n1**4*(-7 + 2*n2) - 16*n1**3*(-68 + 19*n2 + 6*n2**2) - 8*n1**2*(-41 + 568*n2 - 342*n2**2 + 12*n2**3) + 4*n1*(-23 + 1814*n2 - 1136*n2**2 - 76*n2**3 + 8*n2**4)))/(288.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_84(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(16*n1**3*(-6 + 5*n2 + 2*n2**2) + 8*n1**2*(-57 + 104*n2 - 90*n2**2 + 4*n2**3) - 3*(-75 + 30*n2 + 152*n2**2 + 32*n2**3) + n1*(-90 - 232*n2 + 832*n2**2 + 80*n2**3)))/(288.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_85(n1, n2, f):
            return (f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(5 - 6*n2 + n1*(-6 + 4*n2)))/(96.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_86(n1, n2, f):
            return (-15*f**2*(-5 + 2*n1)*(-3 + 2*n1)*(-5 + 2*n2)*(-3 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_87(n1, n2, f):
            return (-9*f**2*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(135 + 28*n1**2 + n1*(8 - 84*n2) + 8*n2 + 28*n2**2))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_88(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(3885 + 358*n2 - 2312*n2**2 - 1792*n2**3 - 352*n2**4 + 32*n1**4*(-11 + 21*n2) - 16*n1**3*(112 - 23*n2 + 84*n2**2) - 8*n1**2*(289 - 546*n2 - 390*n2**2 + 168*n2**3) + n1*(358 - 7864*n2 + 4368*n2**2 + 368*n2**3 + 672*n2**4)))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_89(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-1815 - 772*n2 + 1764*n2**2 + 16*n2**3 + 16*n1**3*(1 - 25*n2 + 14*n2**2) + 4*n1**2*(441 - 62*n2 - 172*n2**2 + 56*n2**3) - 4*n1*(193 - 210*n2 + 62*n2**2 + 100*n2**3)))/(224.*n1*(1 + n1)*n2*(1 + n2)*(-5 + n1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_90(n1, n2, f):
            return (3*f**2*(18 - 12*n1**3 + 4*n1**4 + 21*n2 - 13*n2**2 - 12*n2**3 + 4*n2**4 + n1*(21 + 2*n2 - 4*n2**2) + n1**2*(-13 - 4*n2 + 8*n2**2)))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_91(n1, n2, f):
            return (f**2*(-3 + 2*n1 + 2*n2)*(-6 + 29*n2 + 21*n2**2 - 14*n2**3 + 2*n1**3*(-7 + 8*n2) + n1**2*(21 - 10*n2 - 16*n2**2) + n1*(29 - 42*n2 - 10*n2**2 + 16*n2**3)))/(16.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_92(n1, n2, f):
            return (f**2*(54 - 81*n2 - 87*n2**2 + 60*n2**3 + 12*n2**4 + 4*n1**4*(3 - 16*n2 + 16*n2**2) + 4*n1**3*(15 + 8*n2 - 80*n2**2 + 32*n2**3) + n1*(-81 - 282*n2 + 420*n2**2 + 32*n2**3 - 64*n2**4) + n1**2*(-87 + 420*n2 + 88*n2**2 - 320*n2**3 + 64*n2**4)))/(32.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_93(n1, n2, f):
            return -0.006944444444444444*(f*(2106 - 1173*n2 - 730*n2**2 + 468*n2**3 - 56*n2**4 + 8*n1**4*(-7 + 2*n2) + 4*n1**3*(117 - 68*n2 + 4*n2**2) + 2*n1**2*(-365 + 348*n2 - 80*n2**2 + 8*n2**3) + n1*(-1173 + 216*n2 + 696*n2**2 - 272*n2**3 + 16*n2**4)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_94(n1, n2, f):
            return -0.013888888888888888*(f*(8*n1**4*(-9 + 4*n2 + 4*n2**2) + 4*n1**3*(63 + 36*n2 - 92*n2**2 + 16*n2**3) - 9*(66 - 77*n2 + 46*n2**2 - 28*n2**3 + 8*n2**4) + 2*n1**2*(-207 - 298*n2 + 528*n2**2 - 184*n2**3 + 16*n2**4) + n1*(693 + 120*n2 - 596*n2**2 + 144*n2**3 + 32*n2**4)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_95(n1, n2, f):
            return -0.020833333333333332*(f*(-3 + 2*n1)*(-3 + 2*n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-6 - n2 + n1*(-1 + 4*n2)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_96(n1, n2, f):
            return (3*f*(-3 + 2*n1)*(-3 + 2*n2)*(-10 - 3*n1 + 2*n1**2 - 3*n2 + 2*n2**2))/(16.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_97(n1, n2, f):
            return -0.017857142857142856*(f*(-3 + 2*n1 + 2*n2)*(606 - 19*n2 - 54*n2**2 - 44*n2**3 + n1**3*(-44 + 84*n2) - 2*n1**2*(27 - 62*n2 + 84*n2**2) + n1*(-19 - 376*n2 + 124*n2**2 + 84*n2**3)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_98(n1, n2, f):
            return -0.008928571428571428*(f*(2754 - 4221*n2 + 348*n2**2 + 900*n2**3 - 48*n2**4 + 16*n1**4*(-3 - 15*n2 + 28*n2**2) + 4*n1**3*(225 - 348*n2 - 404*n2**2 + 224*n2**3) + n1*(-4221 + 276*n2 + 5320*n2**2 - 1392*n2**3 - 240*n2**4) + 4*n1**2*(87 + 1330*n2 - 588*n2**2 - 404*n2**3 + 112*n2**4)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_99(n1, n2, f):
            return 0.024691358024691357 - (27 + 4*n1**2 + 12*n1*(-2 + n2) - 24*n2 + 4*n2**2)/(54.*n1*n2*(-3 + n1 + n2)) + (8*n1**4 + 48*n1**3*(-2 + n2) + (3 - 2*n2)**2*(43 - 18*n2 + 2*n2**2) + n1**2*(406 - 404*n2 + 88*n2**2) + n1*(-678 + 970*n2 - 404*n2**2 + 48*n2**3))/(48.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_100(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(-99 + 216*n2 - 90*n2**2 + 4*n1**3*n2*(-5 + 4*n2) + n1*(216 - 514*n2 + 276*n2**2 - 20*n2**3) + 2*n1**2*(-45 + 138*n2 - 88*n2**2 + 8*n2**3)))/(216.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_101(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(99 - 78*n2 + 18*n2**2 + n1*(-78 + 66*n2 - 20*n2**2) + 2*n1**2*(9 - 10*n2 + 4*n2**2)))/(144.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_102(n1, n2, f):
            return -0.013888888888888888*((-3 + 2*n1)*(-3 + 2*n2)*(-223 + 123*n2 - 14*n2**2 + 2*n1**2*(-7 + 2*n2) + n1*(123 - 62*n2 + 4*n2**2)))/(n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_103(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(2679 - 2363*n2 + 672*n2**2 - 136*n2**3 + 8*n1**3*(-17 + 5*n2 + 7*n2**2) + n1*(-2363 + 1030*n2 + 298*n2**2 + 40*n2**3) + n1**2*(672 + 298*n2 - 648*n2**2 + 56*n2**3)))/(252.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_104(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(135 - 1011*n2 + 666*n2**2 - 48*n2**3 + 16*n1**3*(-3 - 9*n2 + 7*n2**2) + 2*n1**2*(333 - 78*n2 - 172*n2**2 + 56*n2**3) - n1*(1011 - 1234*n2 + 156*n2**2 + 144*n2**3)))/(168.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_105(n1, n2, f):
            return (3*(-3 + 2*n1)*(11 + 2*n1*(-2 + n2) - 4*n2)*(-3 + 2*n2))/(16.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_106(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(825 + 28*n1**3*(-1 + n2) - 632*n2 + 120*n2**2 - 28*n2**3 + n1**2*(120 + 60*n2 - 112*n2**2) + n1*(-632 + 266*n2 + 60*n2**2 + 28*n2**3)))/(56.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))

        def M22direct_107(n1, n2, f):
            return ((-3 + 2*n1 + 2*n2)*(-8307 + 11196*n2 - 960*n2**2 + 72*n2**3 + 8*n1**3*(9 - 7*n2 + 98*n2**2) - 2*n1*(-5598 + 911*n2 + 1936*n2**2 + 28*n2**3) + 8*n1**2*(-120 - 484*n2 - 63*n2**2 + 98*n2**3)))/(784.*n1*(1 + n1)*n2*(1 + n2)*(-4 + n1 + n2)*(-3 + n1 + n2))




        ffid = self.scale_independent_growth_factor_f(z)
        # cdef int Nmax = 256
        cdef double b = -1.7        
        # cdef double k0 = 1.039554593366e-05
        # cdef double kmax = 106.1391305637
        cdef double k0 = 5.0e-05
        cdef double kmax = 1.0e2

        h= self.ba.h
        Delta = log(kmax/k0) / (Nmax - 1)
        indices = np.arange(Nmax)
        kbins3 = k0 * np.exp(Delta * indices)
        all_theory = self.get_pk_mult(kbins3*h,z,len(kbins3))
        Plin0 = all_theory[14]*h**3.
        Plin = interpolate.InterpolatedUnivariateSpline(kbins3,Plin0,ext=3)
        Pdiscrin = Plin(k0 * np.exp(Delta * indices)) * np.exp(-1. * b * indices * Delta)
        jsNm = np.arange(-Nmax/2,Nmax/2+1,1)
        etam = b + 2*1j*np.pi*(jsNm)/Nmax/Delta
        cm = np.fft.fft(Pdiscrin)/ Nmax
        cmsym = np.zeros(Nmax+1, dtype=complex)

        idx = np.arange(Nmax+1,dtype=np.intc)
        cond = (idx + 2 - Nmax//2) < 1
        int_indices = (-idx[cond] + Nmax //2).astype(np.intc)
        cmsym[cond] = k0 ** (-etam[cond]) * np.conjugate(cm[int_indices])
        cmsym[~cond] = k0 ** (-etam[~cond]) * cm[idx[~cond] - Nmax//2]
            
        cmsym[-1] = cmsym[-1] / 2
        cmsym[0] = cmsym[0] / 2

        matextR = np.loadtxt('/home1/mengxiwu/public_class_pt_v2/python/new_loops/genM22_basic_oneline_real_N'+str(Nmax)+'.dat')
        matextI = np.loadtxt('/home1/mengxiwu/public_class_pt_v2/python/new_loops/genM22_basic_oneline_imag_N'+str(Nmax)+'.dat')

        m22master = np.zeros((Nmax+1,Nmax+1),dtype=complex)
        for i in range(Nmax+1):
            for j in range(Nmax+1):
                m22master[i][j] = matextR[i * (Nmax+1)+j] +1j* matextI[i * (Nmax+1)+j];

        m13in = np.loadtxt('/home1/mengxiwu/public_class_pt_v2/python/new_loops/genM13_basic_oneline_N'+str(Nmax)+'.dat')

        m13new = np.zeros((Nmax+1),dtype=complex)
        for i in range(Nmax+1):
            m13new[i] = m13in[i] + 1j*m13in[i+Nmax+1]

        m13direct_functions = {
            0: M13direct_0,
    1: M13direct_1,
    2: M13direct_2,
    3: M13direct_3,
    4: M13direct_4,
    5: M13direct_5,
    6: M13direct_6,
    7: M13direct_7,
    8: M13direct_8,
    9: M13direct_9,
    10: M13direct_10,
    11: M13direct_11,
    12: M13direct_12,
    13: M13direct_13,
    14: M13direct_14,
    15: M13direct_15,
    16: M13direct_16,
    17: M13direct_17,
    18: M13direct_18,
    19: M13direct_19,
    20: M13direct_20,
    21: M13direct_21,
    22: M13direct_22,
    23: M13direct_23,
    24: M13direct_24,
    25: M13direct_25,
    26: M13direct_26
        }


        m22direct_functions = {0: M22direct_0,
    1: M22direct_1,
    2: M22direct_2,
    3: M22direct_3,
    4: M22direct_4,
    5: M22direct_5,
    6: M22direct_6,
    7: M22direct_7,
    8: M22direct_8,
    9: M22direct_9,
    10: M22direct_10,
    11: M22direct_11,
    12: M22direct_12,
    13: M22direct_13,
    14: M22direct_14,
    15: M22direct_15,
    16: M22direct_16,
    17: M22direct_17,
    18: M22direct_18,
    19: M22direct_19,
    20: M22direct_20,
    21: M22direct_21,
    22: M22direct_22,
    23: M22direct_23,
    24: M22direct_24,
    25: M22direct_25,
    26: M22direct_26,
    27: M22direct_27,
    28: M22direct_28,
    29: M22direct_29,
    30: M22direct_30,
    31: M22direct_31,
    32: M22direct_32,
    33: M22direct_33,
    34: M22direct_34,
    35: M22direct_35,
    36: M22direct_36,
    37: M22direct_37,
    38: M22direct_38,
    39: M22direct_39,
    40: M22direct_40,
    41: M22direct_41,
    42: M22direct_42,
    43: M22direct_43,
    44: M22direct_44,
    45: M22direct_45,
    46: M22direct_46,
    47: M22direct_47,
    48: M22direct_48,
    49: M22direct_49,
    50: M22direct_50,
    51: M22direct_51,
    52: M22direct_52,
    53: M22direct_53,
    54: M22direct_54,
    55: M22direct_55,
    56: M22direct_56,
    57: M22direct_57,
    58: M22direct_58,
    59: M22direct_59,
    60: M22direct_60,
    61: M22direct_61,
    62: M22direct_62,
    63: M22direct_63,
    64: M22direct_64,
    65: M22direct_65,
    66: M22direct_66,
    67: M22direct_67,
    68: M22direct_68,
    69: M22direct_69,
    70: M22direct_70,
    71: M22direct_71,
    72: M22direct_72,
    73: M22direct_73,
    74: M22direct_74,
    75: M22direct_75,
    76: M22direct_76,
    77: M22direct_77,
    78: M22direct_78,
    79: M22direct_79,
    80: M22direct_80,
    81: M22direct_81,
    82: M22direct_82,
    83: M22direct_83,
    84: M22direct_84,
    85: M22direct_85,
    86: M22direct_86,
    87: M22direct_87,
    88: M22direct_88,
    89: M22direct_89,
    90: M22direct_90,
    91: M22direct_91,
    92: M22direct_92,
    93: M22direct_93,
    94: M22direct_94,
    95: M22direct_95,
    96: M22direct_96,
    97: M22direct_97,
    98: M22direct_98,
    99: M22direct_99,
    100: M22direct_100,
    101: M22direct_101,
    102: M22direct_102,
    103: M22direct_103,
    104: M22direct_104,
    105: M22direct_105,
    106: M22direct_106,
    107: M22direct_107
        }


        n1,n2 = np.meshgrid(-0.5*etam, -0.5*etam)
        m22mat = np.empty((Nmax+1, Nmax+1), dtype=complex)
        P22 = np.zeros(Nmax)
        modelvec22 = kbins3
        for i in range(108):
            index = i
            m22extra = m22direct_functions[i](n1,n2,ffid)
            m22mat = m22master * m22extra
            for j in range (Nmax):
                b1 = cmsym * kbins3[j]**(etam)
                f22out = np.dot(b1,np.dot(m22mat,b1))
                P22[j] = np.real(kbins3[j]**3 * f22out)
            modelvec22 = np.column_stack([modelvec22, P22])
        P13 = np.zeros(Nmax)
        modelvec13 = kbins3
        m13 = np.zeros((Nmax+1),dtype=complex)
        b13 = np.zeros((Nmax+1),dtype=complex)
        modelvec = modelvec22

        for i in range(27):
            index = i
            m13extra = m13direct_functions[i](-0.5*etam, ffid)
            m13 = m13extra*m13new
            for j in range(Nmax):
                b13 = cmsym * kbins3[j]**(etam)
                f13 = np.dot(b13,m13)
                P13[j] = np.real(kbins3[j]**3 * Plin(kbins3[j]) * f13)
            modelvec = np.column_stack([modelvec, P13])
            modelvec13 = np.column_stack([modelvec13, P13]) 
        return modelvec

    def get_pk_comp(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """Fast function to get the non-linear power spectrum components on a k and z array"""
        #cdef np.ndarray[DTYPE_t, ndim = 4] pk_mult = np.zeros((72,k_size,z_size,mu_size),'float64')
        #GC: ORTHOGONAL -- start
        cdef np.ndarray[DTYPE_t, ndim = 4] pk_mult = np.zeros((96,k_size,z_size,mu_size),'float64')
        #GC: ORTHOGONAL -- finish
        cdef np.ndarray[DTYPE_t, ndim = 1] this_pk
        cdef int index_k, index_z, index_mu, index_comb

        for index_k in xrange(k_size):
            for index_z in xrange(z_size):
                for index_mu in xrange(mu_size):
                    this_pk = np.asarray(self.pk(k[index_k,index_z,index_mu],z[index_z]))
                    for index_comb in xrange(96): #GC: ORTHOGONAL...
                        pk_mult[index_comb, index_k, index_z, index_mu] = this_pk[index_comb]

        return pk_mult

    def get_pk_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """ Fast function to get the linear power spectrum on a k and z array """
        cdef np.ndarray[DTYPE_t, ndim=3] pk = np.zeros((k_size,z_size,mu_size),'float64')
        cdef int index_k, index_z, index_mu

        for index_k in xrange(k_size):
            for index_z in xrange(z_size):
                for index_mu in xrange(mu_size):
                    pk[index_k,index_z,index_mu] = self.pk_lin(k[index_k,index_z,index_mu],z[index_z])
        return pk

    # New user interface functions
    def initialize_output(self, np.ndarray[DTYPE_t,ndim=1] k, double z, int k_size):
        """Compute and store the various non-linear power spectrum terms that will be called in the user interface routines below."""
        self.kh = k
        self.fz = self.scale_independent_growth_factor_f(z)
        self.pk_mult = self.get_pk_mult(k, z, k_size)
        self.output_init = True

    def pk_mm_real(self, cs):
        """Return real-space matter-matter power spectrum with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[0]+self.pk_mult[14]+2*cs*self.pk_mult[10]/h**2.)*h**3.

    def pk_gg_real(self, b1, b2, bG2, bGamma3, cs, cs0, Pshot):
        """Return real-space galaxy-galaxy power spectrum with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (b1**2.*self.pk_mult[14] + b1**2.*self.pk_mult[0] + 2.*(cs*b1**2.+cs0*b1)*self.pk_mult[10]/h**2. + b1*b2*self.pk_mult[2]+ 0.25*b2**2.*self.pk_mult[1] + 2.*b1*bG2*self.pk_mult[3] + b1*(2.*bG2 + 0.8*bGamma3)*self.pk_mult[6] + bG2**2.*self.pk_mult[5] + b2*bG2*self.pk_mult[4])*h**3. + Pshot

    def pk_gm_real(self, b1, b2, bG2, bGamma3, cs, cs0):
        """Return real-space galaxy-matter power spectrum with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (b1*self.pk_mult[14] + b1*self.pk_mult[0] + (2.*cs*b1+cs0)*self.pk_mult[10]/h**2. +(b2/2)*self.pk_mult[2] + bG2*self.pk_mult[3] + (bG2 + 0.4*bGamma3)*self.pk_mult[6])*h**3.

#GC -> start...

    def pk_mm_fNL_real(self):
        """Return real-space matter-matter power spectrum fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return self.pk_mult[48]*h**3.


    #GC: ORTHOGONAL -- start
    def pk_mm_fNL_real_ortho(self):
        """Return real-space matter-matter power spectrum fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return self.pk_mult[48+24]*h**3.
    #GC: ORTHOGONAL -- finish


    def pk_gm_fNL_real(self, b1, b2, bG2):
        """Return real-space galaxy-matter power spectrum fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return b1*self.pk_mult[48]*h**3. + ((b2/2.)*self.pk_mult[49]+bG2*self.pk_mult[50])*h**3.


    #GC: ORTHOGONAL -- start
    def pk_gm_fNL_real_ortho(self, b1, b2, bG2):
        """Return real-space galaxy-matter power spectrum fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return b1*self.pk_mult[48+24]*h**3. + ((b2/2.)*self.pk_mult[49+24]+bG2*self.pk_mult[50+24])*h**3.
    #GC: ORTHOGONAL -- finish


    def pk_gg_fNL_real(self, b1, b2, bG2):
        """Return real-space galaxy-galaxy power spectrum fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return b1**2.*self.pk_mult[48]*h**3. + b1*((b2/2.)*self.pk_mult[49]+bG2*self.pk_mult[50])*h**3.


    #GC: ORTHOGONAL -- start
    def pk_gg_fNL_real_ortho(self, b1, b2, bG2):
        """Return real-space galaxy-galaxy power spectrum fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return b1**2.*self.pk_mult[48+24]*h**3. + b1*((b2/2.)*self.pk_mult[49+24]+bG2*self.pk_mult[50+24])*h**3.
    #GC: ORTHOGONAL -- finish


#GC -> end...

    def pk_mm_l0(self, cs0):
        """Return redshift-space matter-matter power spectrum monopole with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[15] + self.pk_mult[21] + self.pk_mult[16] + self.pk_mult[22] + self.pk_mult[17] + self.pk_mult[23] + 2.*cs0*self.pk_mult[11]/h**2.)*h**3.

    def pk_mm_l2(self, cs2):
        """Return redshift-space matter-matter power spectrum quadrupole with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[18] +self.pk_mult[24]+self.pk_mult[19] +self.pk_mult[25] +self.pk_mult[26]  +2.*cs2*self.pk_mult[12]/h**2.)*h**3.

    def pk_mm_l4(self, cs4):
        """Return redshift-space matter-matter power spectrum hexadecapole with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[20] +self.pk_mult[27]+self.pk_mult[28] +self.pk_mult[29] +2.*cs4*self.pk_mult[13]/h**2.)*h**3.

    def pk_gg_l0(self, b1, b2, bG2, bGamma3, cs0, Pshot, b4):
        """Return redshift-space galaxy-galaxy power spectrum monopole with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[15] +self.pk_mult[21]+ b1*self.pk_mult[16] + b1*self.pk_mult[22] + b1**2.*self.pk_mult[17] + b1**2.*self.pk_mult[23] + 0.25*b2**2.*self.pk_mult[1] + b1*b2*self.pk_mult[30]+ b2*self.pk_mult[31] + b1*bG2*self.pk_mult[32] + bG2*self.pk_mult[33]+ b2*bG2*self.pk_mult[4]+ bG2**2.*self.pk_mult[5] + 2.*cs0*self.pk_mult[11]/h**2.
                  + (2.*bG2+0.8*bGamma3)*(b1*self.pk_mult[7]+self.pk_mult[8]))*h**3.+ Pshot + self.fz**2.*b4*(self.kh/h)**2.*(self.fz**2./9. + 2.*self.fz*b1/7. + b1**2./5)*(35./8.)*self.pk_mult[13]*h

    def pk_gg_l2(self, b1, b2, bG2, bGamma3, cs2, b4):
        """Return redshift-space galaxy-galaxy power spectrum quadrupole with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[18] +self.pk_mult[24]+b1*self.pk_mult[19] +b1*self.pk_mult[25] +b1**2.*self.pk_mult[26] +b1*b2*self.pk_mult[34]+b2*self.pk_mult[35] +b1*bG2*self.pk_mult[36]+bG2*self.pk_mult[37]+2.*cs2*self.pk_mult[12]/h**2. +(2.*bG2+0.8*bGamma3)*self.pk_mult[9])*h**3. +self.fz**2.*b4*(self.kh/h)**2.*((self.fz**2.*70. + 165.*self.fz*b1+99.*b1**2.)*4./693.)*(35./8.)*self.pk_mult[13]*h

    def pk_gg_l4(self, b1, b2, bG2, bGamma3, cs4, b4):
        """Return redshift-space galaxy-galaxy power spectrum quadrupole with non-linear corrections. NB: this outputs in (h/Mpc)^3 units"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[20] +self.pk_mult[27]+b1*self.pk_mult[28] +b1**2.*self.pk_mult[29] +b2*self.pk_mult[38] +bG2*self.pk_mult[39] +2.*cs4*self.pk_mult[13]/h**2.)*h**3.+self.fz**2.*b4*(self.kh/h)**2.*((self.fz**2.*210. + 390.*self.fz*b1+143.*b1**2.)*8./5005.)*(35./8.)*self.pk_mult[13]*h

    #GC: I can add something here... E.g. only the pieces from primordial non-Gaussianity, for mm and gg. For mm just take b1=1: this is exactly what Misha does...

    def pk_mm_fNL_l0(self):
        """Return matter-matter power spectrum monopole fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        #return (self.pk_mult[51] + self.pk_mult[52] + self.pk_mult[53])*h**3.
        return (self.pk_mult[51])*h**3.

    def pk_mm_fNL_l2(self):
        """Return matter-matter power spectrum quadrupole fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        #return (self.pk_mult[54] + self.pk_mult[55] + self.pk_mult[56])*h**3.
        return (self.pk_mult[54])*h**3.

    def pk_mm_fNL_l4(self):
        """Return matter-matter power spectrum hexadecapole fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        #return (self.pk_mult[57] + self.pk_mult[58] + self.pk_mult[59])*h**3.
        return (self.pk_mult[57])*h**3.

    def pk_gg_fNL_l0(self, b1, b2, bG2):
        """Return galaxy-galaxy power spectrum monopole fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[51] + b1*self.pk_mult[52] + b1**2.*self.pk_mult[53] + b1*(b2/2.)*self.pk_mult[60] + (b2/2.)*self.pk_mult[61] + b1*bG2*self.pk_mult[62] + bG2*self.pk_mult[63])*h**3.


    def pk_gg_fNL_l2(self, b1, b2, bG2):
        """Return galaxy-galaxy power spectrum quadrupole fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[54] + b1*self.pk_mult[55] + b1**2.*self.pk_mult[56] + b1*(b2/2.)*self.pk_mult[64] + (b2/2.)*self.pk_mult[65] + b1*bG2*self.pk_mult[66] + bG2*self.pk_mult[67])*h**3.


    def pk_gg_fNL_l4(self, b1, b2, bG2):
        """Return galaxy-galaxy power spectrum hexadecapole fNL contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[57] + b1*self.pk_mult[58] + b1**2.*self.pk_mult[59] + b1*(b2/2.)*self.pk_mult[68] + (b2/2.)*self.pk_mult[69] + b1*bG2*self.pk_mult[70] + bG2*self.pk_mult[71])*h**3.



    #GC: ORTHOGONAL -- start



    def pk_mm_fNL_l0_ortho(self):
        """Return matter-matter power spectrum monopole fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        #return (self.pk_mult[51] + self.pk_mult[52] + self.pk_mult[53])*h**3.
        return (self.pk_mult[51+24])*h**3.

    def pk_mm_fNL_l2_ortho(self):
        """Return matter-matter power spectrum quadrupole fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        #return (self.pk_mult[54] + self.pk_mult[55] + self.pk_mult[56])*h**3.
        return (self.pk_mult[54+24])*h**3.

    def pk_mm_fNL_l4_ortho(self):
        """Return matter-matter power spectrum hexadecapole fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        #return (self.pk_mult[57] + self.pk_mult[58] + self.pk_mult[59])*h**3.
        return (self.pk_mult[57+24])*h**3.

    def pk_gg_fNL_l0_ortho(self, b1, b2, bG2):
        """Return galaxy-galaxy power spectrum monopole fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[51+24] + b1*self.pk_mult[52+24] + b1**2.*self.pk_mult[53+24] + b1*(b2/2.)*self.pk_mult[60+24] + (b2/2.)*self.pk_mult[61+24] + b1*bG2*self.pk_mult[62+24] + bG2*self.pk_mult[63+24])*h**3.


    def pk_gg_fNL_l2_ortho(self, b1, b2, bG2):
        """Return galaxy-galaxy power spectrum quadrupole fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[54+24] + b1*self.pk_mult[55+24] + b1**2.*self.pk_mult[56+24] + b1*(b2/2.)*self.pk_mult[64+24] + (b2/2.)*self.pk_mult[65+24] + b1*bG2*self.pk_mult[66+24] + bG2*self.pk_mult[67+24])*h**3.


    def pk_gg_fNL_l4_ortho(self, b1, b2, bG2):
        """Return galaxy-galaxy power spectrum hexadecapole fNL orthogonal contribution. NB: output is in (h/Mpc)^3 units, and one still needs to multiply by fNL"""
        h = self.ba.h
        if not self.output_init:
            raise Exception("Must run initialize_output() before calling this function")
        return (self.pk_mult[57+24] + b1*self.pk_mult[58+24] + b1**2.*self.pk_mult[59+24] + b1*(b2/2.)*self.pk_mult[68+24] + (b2/2.)*self.pk_mult[69+24] + b1*bG2*self.pk_mult[70+24] + bG2*self.pk_mult[71+24])*h**3.



#GC: ORTHOGONAL -- finish



    # Gives sigma(R,z) for a given (R,z)
    def sigma(self,double R,double z):
        """
        Gives the pk for a given R and z
        (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z)

        .. note::

            there is an additional check to verify whether output contains `mPk`,
            and whether k_max > ...
            because otherwise a segfault will occur

        """
        cdef double sigma

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError(
                "No power spectrum computed. In order to get sigma(R,z) you must add mPk to the list of outputs."
                )

        if (self.pt.k_max_for_pk < self.ba.h):
            raise CosmoSevereError(
                "In order to get sigma(R,z) you must set 'P_k_max_h/Mpc' to 1 or bigger, in order to have k_max > 1 h/Mpc."
                )

        if spectra_sigma(&self.ba,&self.pm,&self.sp,R,z,&sigma)==_FAILURE_:
                 raise CosmoSevereError(self.sp.error_message)

        return sigma

    def age(self):
        self.compute(["background"])
        return self.ba.age

    def h(self):
        return self.ba.h

    def n_s(self):
        return self.pm.n_s

    def A_s(self):
        return self.pm.A_s

    def tau_reio(self):
        return self.th.tau_reio

    # Defined twice ?
    def Omega_m(self):
        return self.ba.Omega0_b+self.ba.Omega0_cdm+self.ba.Omega0_ncdm_tot + self.ba.Omega0_dcdm

    #def Omega_r(self):
    #    return self.ba.Omega0_g+self.ba.Omega0_ur

    def omegabh2(self):
        return self.ba.Omega0_b*self.ba.h**2

    def omegach2(self):
        return self.ba.Omega0_cdm*self.ba.h**2

    def Omega_Lambda(self):
        return self.ba.Omega0_lambda

    def Omega_g(self):
        return self.ba.Omega0_g

    def Omega_b(self):
        return self.ba.Omega0_b

    def omega_b(self):
        return self.ba.Omega0_b * self.ba.h * self.ba.h

    def Neff(self):
        return self.ba.Neff

    def sigma8(self):
        self.compute(["spectra"])
        return self.sp.sigma8

    def rs_drag(self):
        self.compute(["thermodynamics"])
        return self.th.rs_d

    def angular_distance(self, z):
        """
        angular_distance(z)

        Return the angular diameter distance (exactly, the quantity defined by Class
        as index_bg_ang_distance in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        cdef double tau
        cdef int last_index #junk
        cdef double * pvecback

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        if background_tau_of_z(&self.ba,z,&tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        D_A = pvecback[self.ba.index_bg_ang_distance]

        free(pvecback)

        return D_A

    def scale_independent_growth_factor(self, z):
        """
        scale_independent_growth_factor(z)

        Return the scale invariant growth factor D(a) for CDM perturbations
        (exactly, the quantity defined by Class as index_bg_D in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        cdef double tau
        cdef int last_index #junk
        cdef double * pvecback

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        if background_tau_of_z(&self.ba,z,&tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        D = pvecback[self.ba.index_bg_D]

        free(pvecback)

        return D

    def scale_independent_growth_factor_f(self, z):
        """
        scale_independent_growth_factor_f(z)

        Return the scale invariant growth factor f(z)=d ln D / d ln a for CDM perturbations
        (exactly, the quantity defined by Class as index_bg_f in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        cdef double tau
        cdef int last_index #junk
        cdef double * pvecback

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        if background_tau_of_z(&self.ba,z,&tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        f = pvecback[self.ba.index_bg_f]

        free(pvecback)

        return f

    def Hubble(self, z):
        """
        Hubble(z)

        Return the Hubble rate (exactly, the quantity defined by Class as index_bg_H
        in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        cdef double tau
        cdef int last_index #junk
        cdef double * pvecback

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        if background_tau_of_z(&self.ba,z,&tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        H = pvecback[self.ba.index_bg_H]

        free(pvecback)

        return H

    def ionization_fraction(self, z):
        """
        ionization_fraction(z)

        Return the ionization fraction for a given redshift z

        Parameters
        ----------
        z : float
                Desired redshift
        """
        cdef double tau
        cdef int last_index #junk
        cdef double * pvecback
        cdef double * pvecthermo

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))
        pvecthermo = <double*> calloc(self.th.th_size,sizeof(double))

        if background_tau_of_z(&self.ba,z,&tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if thermodynamics_at_z(&self.ba,&self.th,z,self.th.inter_normal,&last_index,pvecback,pvecthermo) == _FAILURE_:
            raise CosmoSevereError(self.th.error_message)

        xe = pvecthermo[self.th.index_th_xe]

        free(pvecback)
        free(pvecthermo)

        return xe

    def baryon_temperature(self, z):
        """
        baryon_temperature(z)

        Give the baryon temperature for a given redshift z

        Parameters
        ----------
        z : float
                Desired redshift
        """
        cdef double tau
        cdef int last_index #junk
        cdef double * pvecback
        cdef double * pvecthermo

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))
        pvecthermo = <double*> calloc(self.th.th_size,sizeof(double))

        if background_tau_of_z(&self.ba,z,&tau)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if background_at_tau(&self.ba,tau,self.ba.long_info,self.ba.inter_normal,&last_index,pvecback)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        if thermodynamics_at_z(&self.ba,&self.th,z,self.th.inter_normal,&last_index,pvecback,pvecthermo) == _FAILURE_:
            raise CosmoSevereError(self.th.error_message)

        Tb = pvecthermo[self.th.index_th_Tb]

        free(pvecback)
        free(pvecthermo)

        return Tb

    def T_cmb(self):
        """
        Return the CMB temperature
        """
        return self.ba.T_cmb

    def Omega0_m(self):
        """
        Return the sum of Omega0 for all non-relativistic components
        """
        return self.ba.Omega0_b+self.ba.Omega0_cdm+self.ba.Omega0_ncdm_tot + self.ba.Omega0_dcdm

    def get_background(self):
        """
        Return an array of the background quantities at all times.

        Parameters
        ----------

        Returns
        -------
        background : dictionary containing background.
        """
        cdef char *titles
        cdef double* data
        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if background_output_titles(&self.ba, titles)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.ba.bt_size

        data = <double*>malloc(sizeof(double)*timesteps*number_of_titles)

        if background_output_data(&self.ba, number_of_titles, data)==_FAILURE_:
            raise CosmoSevereError(self.ba.error_message)

        background = {}

        for i in range(number_of_titles):
            background[names[i]] = np.zeros(timesteps, dtype=np.double)
            for index in range(timesteps):
                background[names[i]][index] = data[index*number_of_titles+i]

        return background

    def get_thermodynamics(self):
        """
        Return the thermodynamics quantities.

        Returns
        -------
        thermodynamics : dictionary containing thermodynamics.
        """
        cdef char *titles
        cdef double* data

        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if thermodynamics_output_titles(&self.ba, &self.th, titles)==_FAILURE_:
            raise CosmoSevereError(self.th.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.th.tt_size

        data = <double*>malloc(sizeof(double)*timesteps*number_of_titles)

        if thermodynamics_output_data(&self.ba, &self.th, number_of_titles, data)==_FAILURE_:
            raise CosmoSevereError(self.th.error_message)

        thermodynamics = {}

        for i in range(number_of_titles):
            thermodynamics[names[i]] = np.zeros(timesteps, dtype=np.double)
            for index in range(timesteps):
                thermodynamics[names[i]][index] = data[index*number_of_titles+i]

        return thermodynamics

    def get_primordial(self):
        """
        Return the primordial scalar and/or tensor spectrum depending on 'modes'.
        'output' must be set to something, e.g. 'tCl'.

        Returns
        -------
        primordial : dictionary containing k-vector and primordial scalar and tensor P(k).
        """
        cdef char *titles
        cdef double* data

        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if primordial_output_titles(&self.pt, &self.pm, titles)==_FAILURE_:
            raise CosmoSevereError(self.pm.error_message)

        tmp = <bytes> titles
        names = tmp.split("\t")[:-1]
        tmp = str(tmp.decode())
        number_of_titles = len(names)
        timesteps = self.pm.lnk_size

        data = <double*>malloc(sizeof(double)*timesteps*number_of_titles)

        if primordial_output_data(&self.pt, &self.pm, number_of_titles, data)==_FAILURE_:
            raise CosmoSevereError(self.pm.error_message)

        primordial = {}

        for i in range(number_of_titles):
            primordial[names[i]] = np.zeros(timesteps, dtype=np.double)
            for index in range(timesteps):
                primordial[names[i]][index] = data[index*number_of_titles+i]

        return primordial


    def get_perturbations(self):
        """
        Return scalar, vector and/or tensor perturbations as arrays for requested
        k-values.

        .. note::

            you need to specify both 'k_output_values', and have some
            perturbations computed, for instance by setting 'output' to 'tCl'.

        Returns
        -------
        perturbations : dict of array of dicts
                perturbations['scalar'] is an array of length 'k_output_values' of
                dictionary containing scalar perturbations.
                Similar for perturbations['vector'] and perturbations['tensor'].
        """

        perturbations = {}

        if self.pt.k_output_values_num<1:
            return perturbations

        # Doing the exact same thing 3 times, for scalar, vector and tensor. Sorry
        # for copy-and-paste here, but I don't know what else to do.

        #Scalar:
        if self.pt.has_scalars:
            tmp = <bytes> self.pt.scalar_titles
            tmp = str(tmp.decode())
            names = tmp.split("\t")[:-1]
            number_of_titles = len(names)
            tmparray = [];
            if number_of_titles != 0:
                for j in range(self.pt.k_output_values_num):
                    timesteps = self.pt.size_scalar_perturbation_data[j]//number_of_titles;
                    tmpdict={}
                    for i in range(number_of_titles):
                        tmpdict[names[i]] = np.zeros(timesteps, dtype=np.double)
                        for index in range(timesteps):
                            tmpdict[names[i]][index] = self.pt.scalar_perturbations_data[j][index*number_of_titles+i]
                    tmparray.append(tmpdict)
            perturbations['scalar'] = tmparray;

        #Vector:
        if self.pt.has_vectors:
            tmp = <bytes> self.pt.vector_titles
            tmp = str(tmp.decode())
            names = tmp.split("\t")[:-1]
            number_of_titles = len(names)
            tmparray = [];
            if number_of_titles != 0:
                for j in range(self.pt.k_output_values_num):
                    timesteps = self.pt.size_vector_perturbation_data[j]//number_of_titles;
                    tmpdict={}
                    for i in range(number_of_titles):
                        tmpdict[names[i]] = np.zeros(timesteps, dtype=np.double)
                        for index in range(timesteps):
                            tmpdict[names[i]][index] = self.pt.vector_perturbations_data[j][index*number_of_titles+i]
                    tmparray.append(tmpdict)
            perturbations['vector'] = tmparray;

        #Tensor:
        if self.pt.has_tensors:
            tmp = <bytes> self.pt.tensor_titles
            tmp = str(tmp.decode())
            names = tmp.split("\t")[:-1]
            number_of_titles = len(names)
            tmparray = [];
            if number_of_titles != 0:
                for j in range(self.pt.k_output_values_num):
                    timesteps = self.pt.size_tensor_perturbation_data[j]//number_of_titles;
                    tmpdict={}
                    for i in range(number_of_titles):
                        tmpdict[names[i]] = np.zeros(timesteps, dtype=np.double)
                        for index in range(timesteps):
                            tmpdict[names[i]][index] = self.pt.tensor_perturbations_data[j][index*number_of_titles+i]
                    tmparray.append(tmpdict)
            perturbations['tensor'] = tmparray;

        return perturbations

    def get_transfer(self, z=0., output_format='class'):
        """
        Return the density and/or velocity transfer functions for all initial
        conditions today. You must include 'dCl' and 'vCl' in the list of
        'output'. The transfer functions can also be computed at higher redshift z
        provided that 'z_pk' has been set and that z is inside the region spanned by 'z_pk'.

        Parameters
        ----------
        z  : redshift (default = 0)
        output_format  : ('class' or 'camb') Format transfer functions according to
                         CLASS convention (default) or CAMB convention.

        Returns
        -------
        tk : dictionary containing transfer functions.
        """
        cdef char *titles
        cdef double* data
        cdef char ic_info[1024]
        cdef FileName ic_suffix
        cdef file_format outf

        if (not self.pt.has_density_transfers) and (not self.pt.has_velocity_transfers):
            return {}

        if output_format == 'camb':
            outf = camb_format
        else:
            outf = class_format

        index_md = 0;
        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if spectra_output_tk_titles(&self.ba,&self.pt, outf, titles)==_FAILURE_:
            raise CosmoSevereError(self.op.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.sp.ln_k_size

        size_ic_data = timesteps*number_of_titles;
        ic_num = self.sp.ic_size[index_md];

        data = <double*>malloc(sizeof(double)*size_ic_data*ic_num)

        if spectra_output_tk_data(&self.ba, &self.pt, &self.sp, outf, <double> z, number_of_titles, data)==_FAILURE_:
            raise CosmoSevereError(self.sp.error_message)

        spectra = {}

        for index_ic in range(ic_num):
            if spectra_firstline_and_ic_suffix(&self.pt, index_ic, ic_info, ic_suffix)==_FAILURE_:
                raise CosmoSevereError(self.op.error_message)
            ic_key = <bytes> ic_suffix

            tmpdict = {}
            for i in range(number_of_titles):
                tmpdict[names[i]] = np.zeros(timesteps, dtype=np.double)
                for index in range(timesteps):
                    tmpdict[names[i]][index] = data[index_ic*size_ic_data+index*number_of_titles+i]

            if ic_num==1:
                spectra = tmpdict
            else:
                spectra[ic_key] = tmpdict

        return spectra


    def get_current_derived_parameters(self, names):
        """
        get_current_derived_parameters(names)

        Return a dictionary containing an entry for all the names defined in the
        input list.

        Parameters
        ----------
        names : list
                Derived parameters that can be asked from Monte Python, or
                elsewhere.

        Returns
        -------
        derived : dict

        .. warning::

            This method used to take as an argument directly the data class from
            Monte Python. To maintain compatibility with this old feature, a
            check is performed to verify that names is indeed a list. If not, it
            returns a TypeError. The old version of this function, when asked
            with the new argument, will raise an AttributeError.

        """
        if type(names) != type([]):
            raise TypeError("Deprecated")

        derived = {}
        for name in names:
            if name == 'h':
                value = self.ba.h
            elif name == 'H0':
                value = self.ba.h*100
            elif name == 'Omega0_lambda' or name == 'Omega_Lambda':
                value = self.ba.Omega0_lambda
            elif name == 'Omega0_fld':
                value = self.ba.Omega0_fld
            elif name == 'age':
                value = self.ba.age
            elif name == 'conformal_age':
                value = self.ba.conformal_age
            elif name == 'm_ncdm_in_eV':
                value = self.ba.m_ncdm_in_eV[0]
            elif name == 'm_ncdm_tot':
                value = self.ba.Omega0_ncdm_tot*self.ba.h*self.ba.h*93.14
            elif name == 'Neff':
                value = self.ba.Neff
            elif name == 'Omega_m':
                value = (self.ba.Omega0_b + self.ba.Omega0_cdm+
                         self.ba.Omega0_ncdm_tot + self.ba.Omega0_dcdm)
            elif name == 'omega_m':
                value = (self.ba.Omega0_b + self.ba.Omega0_cdm+
                         self.ba.Omega0_ncdm_tot + self.ba.Omega0_dcdm)/self.ba.h**2
            elif name == 'tau_reio':
                value = self.th.tau_reio
            elif name == 'z_reio':
                value = self.th.z_reio
            elif name == 'z_rec':
                value = self.th.z_rec
            elif name == 'tau_rec':
                value = self.th.tau_rec
            elif name == 'rs_rec':
                value = self.th.rs_rec
            elif name == 'rs_rec_h':
                value = self.th.rs_rec*self.ba.h
            elif name == 'ds_rec':
                value = self.th.ds_rec
            elif name == 'ds_rec_h':
                value = self.th.ds_rec*self.ba.h
            elif name == 'ra_rec':
                value = self.th.da_rec*(1.+self.th.z_rec)
            elif name == 'ra_rec_h':
                value = self.th.da_rec*(1.+self.th.z_rec)*self.ba.h
            elif name == 'da_rec':
                value = self.th.da_rec
            elif name == 'da_rec_h':
                value = self.th.da_rec*self.ba.h
            elif name == 'z_d':
                value = self.th.z_d
            elif name == 'tau_d':
                value = self.th.tau_d
            elif name == 'ds_d':
                value = self.th.ds_d
            elif name == 'ds_d_h':
                value = self.th.ds_d*self.ba.h
            elif name == 'rs_d':
                value = self.th.rs_d
            elif name == 'rs_d_h':
                value = self.th.rs_d*self.ba.h
            elif name == '100*theta_s':
                value = 100.*self.th.rs_rec/self.th.da_rec/(1.+self.th.z_rec)
            elif name == 'YHe':
                value = self.th.YHe
            elif name == 'n_e':
                value = self.th.n_e
            elif name == 'A_s':
                value = self.pm.A_s
            elif name == 'ln10^{10}A_s':
                value = log(1.e10*self.pm.A_s)
            elif name == 'n_s':
                value = self.pm.n_s
            elif name == 'alpha_s':
                value = self.pm.alpha_s
            elif name == 'beta_s':
                value = self.pm.beta_s
            elif name == 'r':
                # This is at the pivot scale
                value = self.pm.r
            elif name == 'r_0002':
                # at k_pivot = 0.002/Mpc
                value = self.pm.r*(0.002/self.pm.k_pivot)**(
                    self.pm.n_t-self.pm.n_s-1+0.5*self.pm.alpha_s*log(
                        0.002/self.pm.k_pivot))
            elif name == 'n_t':
                value = self.pm.n_t
            elif name == 'alpha_t':
                value = self.pm.alpha_t
            elif name == 'V_0':
                value = self.pm.V0
            elif name == 'V_1':
                value = self.pm.V1
            elif name == 'V_2':
                value = self.pm.V2
            elif name == 'V_3':
                value = self.pm.V3
            elif name == 'V_4':
                value = self.pm.V4
            elif name == 'epsilon_V':
                eps1 = self.pm.r*(1./16.-0.7296/16.*(self.pm.r/8.+self.pm.n_s-1.))
                eps2 = -self.pm.n_s+1.-0.7296*self.pm.alpha_s-self.pm.r*(1./8.+1./8.*(self.pm.n_s-1.)*(-0.7296-1.5))-(self.pm.r/8.)**2*(-0.7296-1.)
                value = eps1*((1.-eps1/3.+eps2/6.)/(1.-eps1/3.))**2
            elif name == 'eta_V':
                eps1 = self.pm.r*(1./16.-0.7296/16.*(self.pm.r/8.+self.pm.n_s-1.))
                eps2 = -self.pm.n_s+1.-0.7296*self.pm.alpha_s-self.pm.r*(1./8.+1./8.*(self.pm.n_s-1.)*(-0.7296-1.5))-(self.pm.r/8.)**2*(-0.7296-1.)
                eps23 = 1./8.*(self.pm.r**2/8.+(self.pm.n_s-1.)*self.pm.r-8.*self.pm.alpha_s)
                value = (2.*eps1-eps2/2.-2./3.*eps1**2+5./6.*eps1*eps2-eps2**2/12.-eps23/6.)/(1.-eps1/3.)
            elif name == 'ksi_V^2':
                eps1 = self.pm.r*(1./16.-0.7296/16.*(self.pm.r/8.+self.pm.n_s-1.))
                eps2 = -self.pm.n_s+1.-0.7296*self.pm.alpha_s-self.pm.r*(1./8.+1./8.*(self.pm.n_s-1.)*(-0.7296-1.5))-(self.pm.r/8.)**2*(-0.7296-1.)
                eps23 = 1./8.*(self.pm.r**2/8.+(self.pm.n_s-1.)*self.pm.r-8.*self.pm.alpha_s)
                value = 2.*(1.-eps1/3.+eps2/6.)*(2.*eps1**2-3./2.*eps1*eps2+eps23/4.)/(1.-eps1/3.)**2
            elif name == 'exp_m_2_tau_As':
                value = exp(-2.*self.th.tau_reio)*self.pm.A_s
            elif name == 'phi_min':
                value = self.pm.phi_min
            elif name == 'phi_max':
                value = self.pm.phi_max
            elif name == 'alpha_kp':
                value = self.sp.alpha_kp
            elif name == 'alpha_k1':
                value = self.sp.alpha_k1
            elif name == 'alpha_k2':
                value = self.sp.alpha_k2
            elif name == 'alpha_II_2_20':
                value = self.sp.alpha_II_2_20
            elif name == 'alpha_RI_2_20':
                value = self.sp.alpha_RI_2_20
            elif name == 'alpha_RR_2_20':
                value = self.sp.alpha_RR_2_20
            elif name == 'alpha_II_21_200':
                value = self.sp.alpha_II_21_200
            elif name == 'alpha_RI_21_200':
                value = self.sp.alpha_RI_21_200
            elif name == 'alpha_RR_21_200':
                value = self.sp.alpha_RR_21_200
            elif name == 'alpha_II_201_2500':
                value = self.sp.alpha_II_201_2500
            elif name == 'alpha_RI_201_2500':
                value = self.sp.alpha_RI_201_2500
            elif name == 'alpha_RR_201_2500':
                value = self.sp.alpha_RR_201_2500
            elif name == 'alpha_II_2_2500':
                value = self.sp.alpha_II_2_2500
            elif name == 'alpha_RI_2_2500':
                value = self.sp.alpha_RI_2_2500
            elif name == 'alpha_RR_2_2500':
                value = self.sp.alpha_RR_2_2500
            elif name == 'sigma8':
                value = self.sp.sigma8
            elif name == 'da_z':
                value = self.sp.sigma8
            else:
                raise CosmoSevereError("%s was not recognized as a derived parameter" % name)
            derived[name] = value
        return derived

    def nonlinear_scale(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size):
        """
        nonlinear_scale(z, z_size)

        Return the nonlinear scale for all the redshift specified in z, of size
        z_size

        Parameters
        ----------
        z : numpy array
                Array of requested redshifts
        z_size : int
                Size of the redshift array
        """
        cdef int index_z
        cdef np.ndarray[DTYPE_t, ndim=1] k_nl = np.zeros(z_size,'float64')
        #cdef double *k_nl

        #k_nl = <double*> calloc(z_size,sizeof(double))
        for index_z in range(z_size):
            if nonlinear_k_nl_at_z(&self.ba,&self.nl,z[index_z],&k_nl[index_z]) == _FAILURE_:
                raise CosmoSevereError(self.nl.error_message)

        return k_nl

    def __call__(self, ctx):
        """
        Function to interface with CosmoHammer

        Parameters
        ----------
        ctx : context
                Contains several dictionaries storing data and cosmological
                information

        """
        data = ctx.get('data')  # recover data from the context

        # If the module has already been called once, clean-up
        if self.state:
            self.struct_cleanup()

        # Set the module to the current values
        self.set(data.cosmo_arguments)
        self.compute(["lensing"])

        # Compute the derived paramter value and store them
        params = ctx.getData()
        self.get_current_derived_parameters(
            data.get_mcmc_parameters(['derived']))
        for elem in data.get_mcmc_parameters(['derived']):
            data.mcmc_parameters[elem]['current'] /= \
                data.mcmc_parameters[elem]['scale']
            params[elem] = data.mcmc_parameters[elem]['current']

        ctx.add('boundary', True)
        # Store itself into the context, to be accessed by the likelihoods
        ctx.add('cosmo', self)
