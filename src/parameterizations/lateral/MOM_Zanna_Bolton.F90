!> Calculates Zanna and Bolton 2020 parameterization
!! Implemented by Perezhogin P.A. Contact: pperezhogin@gmail.com
module MOM_Zanna_Bolton

! This file is part of MOM6. See LICENSE.md for the license.
use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type, &
                              start_group_pass, complete_group_pass
use MOM_coms,          only : reproducing_sum
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_cpu_clock,     only : cpu_clock_id, cpu_clock_begin, cpu_clock_end
use MOM_cpu_clock,     only : CLOCK_MODULE, CLOCK_ROUTINE
use MOM_ANN,           only : ANN_init, ANN_apply, ANN_end, ANN_CS
use MOM_error_handler, only : MOM_mesg
use MOM_spatial_means, only : global_volume_mean

implicit none ; private

#include <MOM_memory.h>

public ZB2020_lateral_stress, ZB2020_init, ZB2020_end, ZB2020_copy_gradient_and_thickness

real, public :: GM_dissipation !< Dissipation by the GM parameterization in units of 
                         !! Watt divided by the density, when the GM diffusivity equals 1 m2/s
real, public :: GM_coefficient !< The GM coefficient which is used to equilibrate the overwhelming
                         ! KE backscatter
logical, public :: GM_conserv  !< If true, adds GM dissipation to equilibrate bakcscatter of KE

!> Control structure for Zanna-Bolton-2020 parameterization.
type, public :: ZB2020_CS ; private
  ! Parameters
  real      :: amplitude      !< The nondimensional scaling factor in ZB model,
                              !! typically 0.1 - 10 [nondim].
  integer   :: ZB_type        !< Select how to compute the trace part of ZB model:
                              !! 0 - both deviatoric and trace components are computed
                              !! 1 - only deviatoric component is computed
                              !! 2 - only trace component is computed
  integer   :: ZB_cons        !< Select a discretization scheme for ZB model
                              !! 0 - non-conservative scheme
                              !! 1 - conservative scheme for deviatoric component
  integer   :: HPF_iter       !< Number of sharpening passes for the Velocity Gradient (VG) components
                              !! in ZB model.
  integer   :: Stress_iter    !< Number of smoothing passes for the Stress tensor components
                              !! in ZB model.
  real      :: Klower_R_diss  !< Attenuation of
                              !! the ZB parameterization in the regions of
                              !! geostrophically-unbalanced flows (Klower 2018, Juricke2020,2019)
                              !! Subgrid stress is multiplied by 1/(1+(shear/(f*R_diss)))
                              !! R_diss=-1: attenuation is not used; typical value R_diss=1.0 [nondim]
  integer   :: Klower_shear   !< Type of expression for shear in Klower formula
                              !! 0: sqrt(sh_xx**2 + sh_xy**2)
                              !! 1: sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)
  integer   :: Marching_halo  !< The number of filter iterations per a single MPI
                              !! exchange
  real      :: DT             !< The (baroclinic) dynamics time step [T ~> s]

  real, dimension(:,:,:), allocatable :: &
          sh_xx,   & !< Horizontal tension (du/dx - dv/dy) in h (CENTER)
                     !! points including metric terms [T-1 ~> s-1]
          sh_xy,   & !< Horizontal shearing strain (du/dy + dv/dx) in q (CORNER)
                     !! points including metric terms [T-1 ~> s-1]
          vort_xy, & !< Vertical vorticity (dv/dx - du/dy) in q (CORNER)
                     !! points including metric terms [T-1 ~> s-1]
          hq         !< Thickness in CORNER points [H ~> m or kg m-2]

  real, dimension(:,:,:), allocatable :: &
          Txx,     & !< Subgrid stress xx component in h [L2 T-2 ~> m2 s-2]
          Tyy,     & !< Subgrid stress yy component in h [L2 T-2 ~> m2 s-2]
          Txy        !< Subgrid stress xy component in q [L2 T-2 ~> m2 s-2]

  real, dimension(:,:), allocatable :: &
          kappa_h, & !< Scaling coefficient in h points [L2 ~> m2]
          kappa_q    !< Scaling coefficient in q points [L2 ~> m2]

  real, allocatable ::    &
        ICoriolis_h(:,:), &  !< Inverse Coriolis parameter at h points [T ~> s]
        c_diss(:,:,:)        !< Attenuation parameter at h points
                             !! (Klower 2018, Juricke2019,2020) [nondim]

  real, dimension(:,:), allocatable ::    &
        maskw_h,  & !< Mask of land point at h points multiplied by filter weight [nondim]
        maskw_q     !< Same mask but for q points [nondim]

  real, dimension(:,:,:), allocatable :: &
        Esource_smag, & !< Subgrid Energy source due to Smagorinsky model [L5 T-3 ~> m5 s-3]
        Esource_ZB      !< Subgrid Energy source due to ZB2020 model [L5 T-3 ~> m5 s-3]

  real :: backscatter_ratio !< The ratio of backscattered energy to the dissipated energy

  integer :: use_ann  !< 0: ANN is turned off, 1: default ANN with ZB20 model, 2: two separate ANNs on stencil 3x3 for corner and center
  logical :: rotation_invariant !< If true, the ANN is rotation invariant
  logical :: ann_smag_conserv !< Energy-conservative ANN by imposing Smagorinsky model
  logical :: smag_conserv_lagrangian !< Energy conservation is imposed by introducing Smagorinsky model and performing averaging in Lagrangian frame
  type(ANN_CS) :: ann_instance !< ANN instance
  type(ANN_CS) :: ann_Txy !< ANN instance for Txy
  type(ANN_CS) :: ann_Txx_Tyy !< ANN instance for diagonal stress
  character(len=200) :: ann_file = "/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_64_neurons_ZB-ver-1.2.nc" !< Default ANN with ZB20 model
  character(len=200) :: ann_file_Txy
  character(len=200) :: ann_file_Txx_Tyy
  real :: subroundoff_shear

  type(diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_ZB2020u = -1, id_ZB2020v = -1, id_KE_ZB2020 = -1
  integer :: id_Txx = -1
  integer :: id_Tyy = -1
  integer :: id_Txy = -1
  integer :: id_cdiss = -1
  integer :: id_h = -1, id_u = -1, id_v = -1
  integer :: id_smag = -1, id_KE_smag = -1, id_KE_ZB = -1
  integer :: id_GM_coef = -1, id_PE_GM = -1
  integer :: id_attenuation = -1
  integer :: id_Txx_smag = -1, id_Txy_smag = -1
  integer :: id_Esrc_smag = -1, id_Esrc_ZB = -1
  integer :: id_smag_coef = -1
  !>@}

  !>@{ CPU time clock IDs
  integer :: id_clock_module
  integer :: id_clock_copy
  integer :: id_clock_cdiss
  integer :: id_clock_stress
  integer :: id_clock_stress_ANN
  integer :: id_clock_divergence
  integer :: id_clock_mpi
  integer :: id_clock_filter
  integer :: id_clock_post
  integer :: id_clock_source
  !>@}

  !>@{ MPI group passes
  type(group_pass_type) :: &
      pass_Tq, pass_Th, &        !< handles for halo passes of Txy and Txx, Tyy
      pass_xx, pass_xy           !< handles for halo passes of sh_xx and sh_xy, vort_xy
  integer :: Stress_halo = -1, & !< The halo size in filter of the stress tensor
             HPF_halo = -1       !< The halo size in filter of the velocity gradient
  !>@}

end type ZB2020_CS

contains

!> Read parameters, allocate and precompute arrays,
!! register diagnosicts used in Zanna_Bolton_2020().
subroutine ZB2020_init(Time, G, GV, US, param_file, diag, CS, use_ZB2020)
  type(time_type),         intent(in)    :: Time       !< The current model time.
  type(ocean_grid_type),   intent(in)    :: G          !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV         !< The ocean's vertical grid structure
  type(unit_scale_type),   intent(in)    :: US         !< A dimensional unit scaling type
  type(param_file_type),   intent(in)    :: param_file !< Parameter file parser structure.
  type(diag_ctrl), target, intent(inout) :: diag       !< Diagnostics structure.
  type(ZB2020_CS),         intent(inout) :: CS         !< ZB2020 control structure.
  logical,                 intent(out)   :: use_ZB2020 !< If true, turns on ZB scheme.

  real :: subroundoff_Cor     ! A negligible parameter which avoids division by zero
                              ! but small compared to Coriolis parameter [T-1 ~> s-1]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  ! This include declares and sets the variable "version".
#include "version_variable.h"
  character(len=40)  :: mdl = "MOM_Zanna_Bolton" ! This module's name.

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  call log_version(param_file, mdl, version, "")

  call get_param(param_file, mdl, "USE_ZB2020", use_ZB2020, &
                 "If true, turns on Zanna-Bolton-2020 (ZB) " //&
                 "subgrid momentum parameterization of mesoscale eddies.", default=.false.)
  if (.not. use_ZB2020) return

  call get_param(param_file, mdl, "BACKSCATTER_RATIO", CS%backscatter_ratio, &
                 "The ratio between backscattered and dissipated energy." //&
                 "-1 means that backscatter is not controlled", &
                 units="nondim", default=-1.)

  call get_param(param_file, mdl, "USE_ANN", CS%use_ann, &
                 "ANN inference of momentum fluxes: 0 off, 1: single ANN 2x2, 2: two ANNs 3x3", default=0)

  call get_param(param_file, mdl, "ANN_SMAG_CONSERV", CS%ann_smag_conserv, &
                 "Smagorinsky model makes SGS parameterization energy-conservative", default=.False.)

  call get_param(param_file, mdl, "SMAG_CONSERV_LAGRANGIAN", CS%smag_conserv_lagrangian, &
                 "Smagorinsky model makes SGS parameterization energy-conservative in lagrangian frame", &
                 default=.False.)

  call get_param(param_file, mdl, "GM_CONSERV", GM_conserv, &
                 "GM model makes parameterization energy-conservative", default=.False.)
  GM_coefficient = 1e-10
  GM_dissipation = 0.

  call get_param(param_file, mdl, "ANN_FILE_TXY", CS%ann_file_Txy, &
                 "ANN parameters for prediction of Txy netcdf input", &
                 default="/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hdn-64-64/model/Txy_epoch_2000.nc")
  
  call get_param(param_file, mdl, "ANN_FILE_TXX_TYY", CS%ann_file_Txx_Tyy, &
                 "ANN parameters for prediction of Txx and Tyy netcdf input", &
                 default="/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hdn-64-64/model/Txx_Tyy_epoch_2000.nc")

  call get_param(param_file, mdl, "ROT_INV", CS%rotation_invariant, &
                 "If true, rotation invariance is imposed as hard constraint", default=.false.)

  call get_param(param_file, mdl, "ZB_SCALING", CS%amplitude, &
                 "The nondimensional scaling factor in ZB model, " //&
                 "typically 0.5-2.5", units="nondim", default=0.5)

  call get_param(param_file, mdl, "ZB_TRACE_MODE", CS%ZB_type, &
                 "Select how to compute the trace part of ZB model:\n" //&
                 "\t 0 - both deviatoric and trace components are computed\n" //&
                 "\t 1 - only deviatoric component is computed\n" //&
                 "\t 2 - only trace component is computed", default=0)

  call get_param(param_file, mdl, "ZB_SCHEME", CS%ZB_cons, &
                 "Select a discretization scheme for ZB model:\n" //&
                 "\t 0 - non-conservative scheme\n" //&
                 "\t 1 - conservative scheme for deviatoric component", default=1)

  call get_param(param_file, mdl, "VG_SHARP_PASS", CS%HPF_iter, &
                "Number of sharpening passes for the Velocity Gradient (VG) components " //&
                "in ZB model.", default=0)

  call get_param(param_file, mdl, "STRESS_SMOOTH_PASS", CS%Stress_iter, &
                 "Number of smoothing passes for the Stress tensor components " //&
                 "in ZB model.", default=0)

  call get_param(param_file, mdl, "ZB_KLOWER_R_DISS", CS%Klower_R_diss, &
                 "Attenuation of " //&
                 "the ZB parameterization in the regions of " //&
                 "geostrophically-unbalanced flows (Klower 2018, Juricke2020,2019). " //&
                 "Subgrid stress is multiplied by 1/(1+(shear/(f*R_diss))):\n" //&
                 "\t R_diss=-1. - attenuation is not used\n\t R_diss= 1. - typical value", &
                 units="nondim", default=-1.)

  call get_param(param_file, mdl, "ZB_KLOWER_SHEAR", CS%Klower_shear, &
                 "Type of expression for shear in Klower formula:\n" //&
                 "\t 0: sqrt(sh_xx**2 + sh_xy**2)\n" //&
                 "\t 1: sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)", &
                 default=1, do_not_log=.not.CS%Klower_R_diss>0)

  call get_param(param_file, mdl, "ZB_MARCHING_HALO", CS%Marching_halo, &
                 "The number of filter iterations per single MPI " //&
                 "exchange", default=4, do_not_log=(CS%Stress_iter==0).and.(CS%HPF_iter==0))

  call get_param(param_file, mdl, "DT", CS%dt, &
                 "The (baroclinic) dynamics time step.", units="s", scale=US%s_to_T, &
                 fail_if_missing=.true.)

  ! Register fields for output from this module.
  CS%diag => diag

  CS%id_ZB2020u = register_diag_field('ocean_model', 'ZB2020u', diag%axesCuL, Time, &
      'Zonal Acceleration from Zanna-Bolton 2020', 'm s-2', conversion=US%L_T2_to_m_s2)
  CS%id_ZB2020v = register_diag_field('ocean_model', 'ZB2020v', diag%axesCvL, Time, &
      'Meridional Acceleration from Zanna-Bolton 2020', 'm s-2', conversion=US%L_T2_to_m_s2)
  CS%id_KE_ZB2020 = register_diag_field('ocean_model', 'KE_ZB2020', diag%axesTL, Time, &
      'Kinetic Energy Source from Horizontal Viscosity', &
      'm3 s-3', conversion=GV%H_to_m*(US%L_T_to_m_s**2)*US%s_to_T)

  CS%id_Txx = register_diag_field('ocean_model', 'Txx', diag%axesTL, Time, &
      'Diagonal term (Txx) in the ZB stress tensor', 'm2 s-2', conversion=US%L_T_to_m_s**2)
  CS%id_Tyy = register_diag_field('ocean_model', 'Tyy', diag%axesTL, Time, &
      'Diagonal term (Tyy) in the ZB stress tensor', 'm2 s-2', conversion=US%L_T_to_m_s**2)
  CS%id_Txy = register_diag_field('ocean_model', 'Txy', diag%axesBL, Time, &
      'Off-diagonal term (Txy) in the ZB stress tensor', 'm2 s-2', conversion=US%L_T_to_m_s**2)
  
  CS%id_Txx_smag = register_diag_field('ocean_model', 'Txx_smag', diag%axesTL, Time, &
      'Diagonal term (Txx) in the Smagorinsky stress tensor', 'm2 s-2', conversion=US%L_T_to_m_s**2)
  CS%id_Txy_smag = register_diag_field('ocean_model', 'Txy_smag', diag%axesBL, Time, &
      'Off-diagonal term (Txy) in the ZB stress tensor', 'm2 s-2', conversion=US%L_T_to_m_s**2)

  CS%id_Esrc_smag = register_diag_field('ocean_model', 'Esrc_smag', diag%axesTL, Time, &
      'Subgrid KE source by Smagorinsky model', 'm3 s-3', conversion=US%L_T_to_m_s**2 * GV%H_to_m)
  CS%id_Esrc_ZB = register_diag_field('ocean_model', 'Esrc_ZB', diag%axesTL, Time, &
      'Subgrid KE source by ZB20 model', 'm3 s-3', conversion=US%L_T_to_m_s**2 * GV%H_to_m)
  CS%id_smag_coef = register_diag_field('ocean_model', 'smag_coef', diag%axesTL, Time, &
      'Local Smagorinsky coefficient', 'nondim')

  if (CS%Klower_R_diss > 0) then
    CS%id_cdiss = register_diag_field('ocean_model', 'c_diss', diag%axesTL, Time, &
        'Klower (2018) attenuation coefficient', 'nondim')
  endif

  CS%id_h = register_diag_field('ocean_model', 'h_ZB', diag%axesTL, Time, &
    'Thickness in ZB module', 'm', conversion=GV%H_to_m)
  CS%id_u = register_diag_field('ocean_model', 'u_ZB', diag%axesCuL, Time, &
    'Zonal velocity in ZB module', 'ms-1', conversion=US%L_T_to_m_s)
  CS%id_v = register_diag_field('ocean_model', 'v_ZB', diag%axesCvL, Time, &
    'Meridional velocity in ZB module', 'ms-1', conversion=US%L_T_to_m_s)

  CS%id_smag = register_diag_field('ocean_model', 'smag_const', diag%axesNull, Time, &
    'Smagorinsky coefficient determined energetically', 'nondim')
  CS%id_KE_smag = register_diag_field('ocean_model', 'KE_smag', diag%axesNull, Time, &
    'Energetic contribution of Smagorinsky integrated', 'm5 s-3', conversion=GV%H_to_m*(US%L_T_to_m_s**2*US%L_to_m**2)*US%s_to_T)
  CS%id_KE_ZB = register_diag_field('ocean_model', 'KE_ZB', diag%axesNull, Time, &
    'Energetic contribution of ZB2020 integrated', 'm5 s-3', conversion=GV%H_to_m*(US%L_T_to_m_s**2*US%L_to_m**2)*US%s_to_T)
  CS%id_GM_coef = register_diag_field('ocean_model', 'GM_coef', diag%axesNull, Time, &
    'GM coefficient determined energetically', 'm2 s-1', conversion=US%L_to_m**2 / US%T_to_s)
  CS%id_PE_GM = register_diag_field('ocean_model', 'PE_GM', diag%axesNull, Time, &
    'Energetic contribution of GM integrated', 'm5 s-3', conversion=GV%H_to_m*(US%L_T_to_m_s**2*US%L_to_m**2)*US%s_to_T)
  CS%id_attenuation = register_diag_field('ocean_model', 'attenuation', diag%axesNull, Time, &
    'Attenuation of large-scale part of SGS model', 'nondim')

  ! Clock IDs
  ! Only module is measured with syncronization. While smaller
  ! parts are measured without - because these are nested clocks.
  CS%id_clock_module = cpu_clock_id('(Ocean Zanna-Bolton-2020)', grain=CLOCK_MODULE)
  CS%id_clock_copy = cpu_clock_id('(ZB2020 copy fields)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_cdiss = cpu_clock_id('(ZB2020 compute c_diss)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_stress = cpu_clock_id('(ZB2020 compute stress)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_stress_ANN = cpu_clock_id('(ZB2020 compute stress ANN)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_divergence = cpu_clock_id('(ZB2020 compute divergence)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_mpi = cpu_clock_id('(ZB2020 filter MPI exchanges)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_filter = cpu_clock_id('(ZB2020 filter no MPI)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_post = cpu_clock_id('(ZB2020 post data)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_source = cpu_clock_id('(ZB2020 compute energy source)', grain=CLOCK_ROUTINE, sync=.false.)

  CS%subroundoff_shear = 1e-30 * US%T_to_s
  if (CS%use_ann == 1) then
    call ANN_init(CS%ann_instance, CS%ann_file)
  elseif (CS%use_ann == 2) then
    call ANN_init(CS%ann_Txy, CS%ann_file_Txy)
    call ANN_init(CS%ann_Txx_Tyy, CS%ann_file_Txx_Tyy)
  endif

  ! Allocate memory
  ! We set the stress tensor and velocity gradient tensor to zero
  ! with full halo because they potentially may be filtered
  ! with marching halo algorithm
  allocate(CS%sh_xx(SZI_(G),SZJ_(G),SZK_(GV)), source=0.)
  allocate(CS%sh_xy(SZIB_(G),SZJB_(G),SZK_(GV)), source=0.)
  allocate(CS%vort_xy(SZIB_(G),SZJB_(G),SZK_(GV)), source=0.)
  allocate(CS%hq(SZIB_(G),SZJB_(G),SZK_(GV)))

  if (CS%smag_conserv_lagrangian) then
    allocate(CS%Esource_smag(SZI_(G),SZJ_(G),SZK_(GV)), source=0.)
    allocate(CS%Esource_ZB(SZI_(G),SZJ_(G),SZK_(GV)), source=0.)
  endif

  allocate(CS%Txx(SZI_(G),SZJ_(G),SZK_(GV)), source=0.)
  allocate(CS%Tyy(SZI_(G),SZJ_(G),SZK_(GV)), source=0.)
  allocate(CS%Txy(SZIB_(G),SZJB_(G),SZK_(GV)), source=0.)
  allocate(CS%kappa_h(SZI_(G),SZJ_(G)))
  allocate(CS%kappa_q(SZIB_(G),SZJB_(G)))

  ! Precomputing the scaling coefficient
  ! Mask is included to automatically satisfy B.C.
  do j=js-2,je+2 ; do i=is-2,ie+2
    CS%kappa_h(i,j) = -CS%amplitude * G%areaT(i,j) * G%mask2dT(i,j)
  enddo; enddo

  do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
    CS%kappa_q(I,J) = -CS%amplitude * G%areaBu(I,J) * G%mask2dBu(I,J)
  enddo; enddo

  if (CS%Klower_R_diss > 0) then
    allocate(CS%ICoriolis_h(SZI_(G),SZJ_(G)))
    allocate(CS%c_diss(SZI_(G),SZJ_(G),SZK_(GV)))

    subroundoff_Cor = 1e-30 * US%T_to_s
    ! Precomputing 1/(f * R_diss)
    do j=js-1,je+1 ; do i=is-1,ie+1
      CS%ICoriolis_h(i,j) = 1. / ((abs(0.25 * ((G%CoriolisBu(I,J) + G%CoriolisBu(I-1,J-1)) &
                          + (G%CoriolisBu(I-1,J) + G%CoriolisBu(I,J-1)))) + subroundoff_Cor) &
                          * CS%Klower_R_diss)
    enddo; enddo
  endif

  if (CS%Stress_iter > 0 .or. CS%HPF_iter > 0 .or. CS%backscatter_ratio > 0.) then
    ! Include 1/16. factor to the mask for filter implementation
    allocate(CS%maskw_h(SZI_(G),SZJ_(G))); CS%maskw_h(:,:) = G%mask2dT(:,:) * 0.0625
    allocate(CS%maskw_q(SZIB_(G),SZJB_(G))); CS%maskw_q(:,:) = G%mask2dBu(:,:) * 0.0625
  endif

  ! Initialize MPI group passes
  if (CS%Stress_iter > 0) then
    ! reduce size of halo exchange accordingly to
    ! Marching halo, number of iterations and the array size
    ! But let exchange width be at least 1
    CS%Stress_halo = max(min(CS%Marching_halo, CS%Stress_iter, &
                             G%Domain%nihalo, G%Domain%njhalo), 1)

    call create_group_pass(CS%pass_Tq, CS%Txy, G%Domain, halo=CS%Stress_halo, &
      position=CORNER)
    call create_group_pass(CS%pass_Th, CS%Txx, G%Domain, halo=CS%Stress_halo)
    call create_group_pass(CS%pass_Th, CS%Tyy, G%Domain, halo=CS%Stress_halo)
  endif

  if (CS%HPF_iter > 0) then
    ! The minimum halo size is 2 because it is requirement for the
    ! outputs of function filter_velocity_gradients
    CS%HPF_halo = max(min(CS%Marching_halo, CS%HPF_iter, &
                          G%Domain%nihalo, G%Domain%njhalo), 2)

    call create_group_pass(CS%pass_xx, CS%sh_xx, G%Domain, halo=CS%HPF_halo)
    call create_group_pass(CS%pass_xy, CS%sh_xy, G%Domain, halo=CS%HPF_halo, &
      position=CORNER)
    call create_group_pass(CS%pass_xy, CS%vort_xy, G%Domain, halo=CS%HPF_halo, &
      position=CORNER)
  endif

end subroutine ZB2020_init

!> Deallocate any variables allocated in ZB_2020_init
subroutine ZB2020_end(CS)
  type(ZB2020_CS), intent(inout) :: CS  !< ZB2020 control structure.

  deallocate(CS%sh_xx)
  deallocate(CS%sh_xy)
  deallocate(CS%vort_xy)
  deallocate(CS%hq)

  deallocate(CS%Txx)
  deallocate(CS%Tyy)
  deallocate(CS%Txy)
  deallocate(CS%kappa_h)
  deallocate(CS%kappa_q)

  if (CS%Klower_R_diss > 0) then
    deallocate(CS%ICoriolis_h)
    deallocate(CS%c_diss)
  endif

  if (CS%Stress_iter > 0 .or. CS%HPF_iter > 0) then
    deallocate(CS%maskw_h)
    deallocate(CS%maskw_q)
  endif

  if (CS%use_ann==1) then
    call ANN_end(CS%ann_instance)
  elseif (CS%use_ann==2) then
    call ANN_end(CS%ann_Txy)
    call ANN_end(CS%ann_Txx_Tyy)
  endif

end subroutine ZB2020_end

!> Save precomputed velocity gradients and thickness
!! from the horizontal eddy viscosity module
!! We save as much halo for velocity gradients as possible
!! In symmetric (preferable) memory model: halo 2 for sh_xx
!! and halo 1 for sh_xy and vort_xy
!! We apply zero boundary conditions to velocity gradients
!! which is required for filtering operations
subroutine ZB2020_copy_gradient_and_thickness(sh_xx, sh_xy, vort_xy, hq, &
                                       G, GV, CS, k)
  type(ocean_grid_type),         intent(in)    :: G      !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)    :: GV     !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(inout) :: CS     !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(in) :: sh_xy       !< horizontal shearing strain (du/dy + dv/dx)
                              !! including metric terms [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(in) :: vort_xy     !< Vertical vorticity (dv/dx - du/dy)
                              !! including metric terms [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(in) :: hq          !< harmonic mean of the harmonic means
                              !! of the u- & v point thicknesses [H ~> m or kg m-2]

  real, dimension(SZI_(G),SZJ_(G)), &
    intent(in) :: sh_xx       !< horizontal tension (du/dx - dv/dy)
                              !! including metric terms [T-1 ~> s-1]

  integer, intent(in) :: k    !< The vertical index of the layer to be passed.

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  call cpu_clock_begin(CS%id_clock_copy)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  do J=js-1,Jeq ; do I=is-1,Ieq
    CS%hq(I,J,k) = hq(I,J)
  enddo; enddo

  ! No physical B.C. is required for
  ! sh_xx in ZB2020. However, filtering
  ! may require BC
  do j=Jsq-1,je+2 ; do i=Isq-1,ie+2
    CS%sh_xx(i,j,k) = sh_xx(i,j) * G%mask2dT(i,j)
  enddo ; enddo

  ! We multiply by mask to remove
  ! implicit dependence on CS%no_slip
  ! flag in hor_visc module
  do J=js-2,Jeq+1 ; do I=is-2,Ieq+1
    CS%sh_xy(I,J,k) = sh_xy(I,J) * G%mask2dBu(I,J)
  enddo; enddo

  do J=js-2,Jeq+1 ; do I=is-2,Ieq+1
    CS%vort_xy(I,J,k) = vort_xy(I,J) * G%mask2dBu(I,J)
  enddo; enddo

  call cpu_clock_end(CS%id_clock_copy)

end subroutine ZB2020_copy_gradient_and_thickness

!> Baroclinic Zanna-Bolton-2020 parameterization, see
!! eq. 6 in https://laurezanna.github.io/files/Zanna-Bolton-2020.pdf
!! We compute the lateral stress tensor according to ZB2020 model
!! and update the acceleration due to eddy viscosity (diffu, diffv)
!! as follows:
!! diffu = diffu + ZB2020u
!! diffv = diffv + ZB2020v
subroutine ZB2020_lateral_stress(u, v, h, diffu, diffv, G, GV, CS, &
                             dx2h, dy2h, dx2q, dy2q)
  type(ocean_grid_type),         intent(in)    :: G  !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)    :: GV !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(inout) :: CS !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in)    :: u  !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in)    :: v  !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                                 intent(in)    :: h  !< Layer thicknesses [H ~> m or kg m-2].

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                        intent(inout) :: diffu   !< Zonal acceleration due to eddy viscosity.
                                                 !! It is updated with ZB closure [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                        intent(inout) :: diffv   !< Meridional acceleration due to eddy viscosity.
                                                 !! It is updated with ZB closure [L T-2 ~> m s-2]

  real, dimension(SZI_(G),SZJ_(G)), intent(in) :: dx2h    !< dx^2 at h points [L2 ~> m2]
  real, dimension(SZI_(G),SZJ_(G)), intent(in) :: dy2h    !< dy^2 at h points [L2 ~> m2]

  real, dimension(SZIB_(G),SZJB_(G)), intent(in) :: dx2q    !< dx^2 at q points [L2 ~> m2]
  real, dimension(SZIB_(G),SZJB_(G)), intent(in) :: dy2q    !< dy^2 at q points [L2 ~> m2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  call cpu_clock_begin(CS%id_clock_module)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Compute attenuation if specified
  call compute_c_diss(G, GV, CS)

  ! Sharpen velocity gradients if specified
  call filter_velocity_gradients(G, GV, CS)

  ! Compute the stress tensor given the
  ! (optionally sharpened) velocity gradients
  if (CS%use_ann==0) then
    call compute_stress(G, GV, CS)
  elseif (CS%use_ann==1) then
    call compute_stress_ANN(G, GV, CS)
  elseif (CS%use_ann==2) then
    call compute_stress_ANN_3x3(G, GV, CS)
  endif

  ! Smooth the stress tensor if specified
  call filter_stress(G, GV, CS)

  if (CS%smag_conserv_lagrangian) then
    call Bound_backscatter_lagrangian(u, v, h, G, GV, CS)
  endif

  ! Update the acceleration due to eddy viscosity (diffu, diffv)
  ! with the ZB2020 lateral parameterization
  call compute_stress_divergence(u, v, h, diffu, diffv,    &
                                 dx2h, dy2h, dx2q, dy2q, &
                                 G, GV, CS)

  call cpu_clock_begin(CS%id_clock_post)
  if (CS%id_Txx>0)       call post_data(CS%id_Txx, CS%Txx, CS%diag)
  if (CS%id_Tyy>0)       call post_data(CS%id_Tyy, CS%Tyy, CS%diag)
  if (CS%id_Txy>0)       call post_data(CS%id_Txy, CS%Txy, CS%diag)

  if (CS%id_cdiss>0)     call post_data(CS%id_cdiss, CS%c_diss, CS%diag)

  if (CS%id_h>0)       call post_data(CS%id_h, h, CS%diag)
  if (CS%id_u>0)       call post_data(CS%id_u, u, CS%diag)
  if (CS%id_v>0)       call post_data(CS%id_v, v, CS%diag)
  call cpu_clock_end(CS%id_clock_post)

  call cpu_clock_end(CS%id_clock_module)

end subroutine ZB2020_lateral_stress

!> Compute the attenuation parameter similarly
!! to Klower2018, Juricke2019,2020: c_diss = 1/(1+(shear/(f*R_diss)))
!! where shear = sqrt(sh_xx**2 + sh_xy**2) or shear = sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)
!! In symmetric memory model, components of velocity gradient tensor
!! should have halo 1 and zero boundary conditions. The result: c_diss having halo 1.
subroutine compute_c_diss(G, GV, CS)
  type(ocean_grid_type),   intent(in)    :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  real :: shear ! Shear in Klower2018 formula at h points [T-1 ~> s-1]

  if (.not. CS%Klower_R_diss > 0) &
    return

  call cpu_clock_begin(CS%id_clock_cdiss)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  do k=1,nz

    ! sqrt(sh_xx**2 + sh_xy**2)
    if (CS%Klower_shear == 0) then
      do j=js-1,je+1 ; do i=is-1,ie+1
        shear = sqrt(CS%sh_xx(i,j,k)**2 + 0.25 * (          &
          (CS%sh_xy(I-1,J-1,k)**2 + CS%sh_xy(I,J  ,k)**2)   &
        + (CS%sh_xy(I-1,J  ,k)**2 + CS%sh_xy(I,J-1,k)**2)   &
        ))
        CS%c_diss(i,j,k) = 1. / (1. + shear * CS%ICoriolis_h(i,j))
      enddo; enddo

    ! sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)
    elseif (CS%Klower_shear == 1) then
      do j=js-1,je+1 ; do i=is-1,ie+1
        shear = sqrt(CS%sh_xx(i,j,k)**2 + 0.25 * (             &
          ((CS%sh_xy(I-1,J-1,k)**2 + CS%vort_xy(I-1,J-1,k)**2) &
        +  (CS%sh_xy(I,J,k)**2     + CS%vort_xy(I,J,k)**2))    &
        + ((CS%sh_xy(I-1,J,k)**2   + CS%vort_xy(I-1,J,k)**2)   &
        +  (CS%sh_xy(I,J-1,k)**2   + CS%vort_xy(I,J-1,k)**2))  &
        ))
        CS%c_diss(i,j,k) = 1. / (1. + shear * CS%ICoriolis_h(i,j))
      enddo; enddo
    endif

  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_cdiss)

end subroutine compute_c_diss

!> Compute stress tensor T =
!! (Txx, Txy;
!!  Txy, Tyy)
!! Which consists of the deviatoric and trace components, respectively:
!! T =   (-vort_xy * sh_xy, vort_xy * sh_xx;
!!         vort_xy * sh_xx,  vort_xy * sh_xy) +
!! 1/2 * (vort_xy^2 + sh_xy^2 + sh_xx^2, 0;
!!        0, vort_xy^2 + sh_xy^2 + sh_xx^2)
!! This stress tensor is multiplied by precomputed kappa=-CS%amplitude * G%area:
!! T -> T * kappa
!! The sign of the stress tensor is such that (neglecting h):
!! (du/dt, dv/dt) = div(T)
!! In symmetric memory model: sh_xy and vort_xy should have halo 1
!! and zero B.C.; sh_xx should have halo 2 and zero B.C.
!! Result: Txx, Tyy, Txy with halo 1 and zero B.C.
subroutine compute_stress(G, GV, CS)
  type(ocean_grid_type),   intent(in)    :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  real :: &
    vort_xy_h, &  ! Vorticity interpolated to h point [T-1 ~> s-1]
    sh_xy_h       ! Shearing strain interpolated to h point [T-1 ~> s-1]

  real :: &
    sh_xx_q       ! Horizontal tension interpolated to q point [T-1 ~> s-1]

  ! Local variables
  real :: sum_sq  ! 1/2*(vort_xy^2 + sh_xy^2 + sh_xx^2) in h point [T-2 ~> s-2]
  real :: vort_sh ! vort_xy*sh_xy in h point [T-2 ~> s-2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  logical :: sum_sq_flag ! Flag to compute trace
  logical :: vort_sh_scheme_0, vort_sh_scheme_1 ! Flags to compute diagonal trace-free part

  call cpu_clock_begin(CS%id_clock_stress)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  sum_sq = 0.
  vort_sh = 0.

  sum_sq_flag = CS%ZB_type /= 1
  vort_sh_scheme_0 = CS%ZB_type /= 2 .and. CS%ZB_cons == 0
  vort_sh_scheme_1 = CS%ZB_type /= 2 .and. CS%ZB_cons == 1

  do k=1,nz

    ! compute Txx, Tyy tensor
    do j=js-1,je+1 ; do i=is-1,ie+1
      ! It is assumed that B.C. is applied to sh_xy and vort_xy
      sh_xy_h = 0.25 * ( (CS%sh_xy(I-1,J-1,k) + CS%sh_xy(I,J,k)) &
                       + (CS%sh_xy(I-1,J,k) + CS%sh_xy(I,J-1,k)) )

      vort_xy_h = 0.25 * ( (CS%vort_xy(I-1,J-1,k) + CS%vort_xy(I,J,k)) &
                         + (CS%vort_xy(I-1,J,k) + CS%vort_xy(I,J-1,k)) )

      if (sum_sq_flag) then
        sum_sq = 0.5 *                          &
          ((vort_xy_h * vort_xy_h               &
           + sh_xy_h * sh_xy_h)                 &
           + CS%sh_xx(i,j,k) * CS%sh_xx(i,j,k)  &
            )
      endif

      if (vort_sh_scheme_0) &
        vort_sh = vort_xy_h * sh_xy_h

      if (vort_sh_scheme_1) then
        ! It is assumed that B.C. is applied to sh_xy and vort_xy
        vort_sh = 0.25 * (                                                      &
          ((G%areaBu(I-1,J-1) * CS%vort_xy(I-1,J-1,k)) * CS%sh_xy(I-1,J-1,k)  + &
           (G%areaBu(I  ,J  ) * CS%vort_xy(I  ,J  ,k)) * CS%sh_xy(I  ,J  ,k)) + &
          ((G%areaBu(I-1,J  ) * CS%vort_xy(I-1,J  ,k)) * CS%sh_xy(I-1,J  ,k)  + &
           (G%areaBu(I  ,J-1) * CS%vort_xy(I  ,J-1,k)) * CS%sh_xy(I  ,J-1,k))   &
          ) * G%IareaT(i,j)
      endif

      ! B.C. is already applied in kappa_h
      CS%Txx(i,j,k) = CS%kappa_h(i,j) * (- vort_sh + sum_sq)
      CS%Tyy(i,j,k) = CS%kappa_h(i,j) * (+ vort_sh + sum_sq)

    enddo ; enddo

    ! Here we assume that Txy is initialized to zero
    if (CS%ZB_type /= 2) then
      do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
        sh_xx_q = 0.25 * ( (CS%sh_xx(i+1,j+1,k) + CS%sh_xx(i,j,k)) &
                         + (CS%sh_xx(i+1,j,k) + CS%sh_xx(i,j+1,k)))
        ! B.C. is already applied in kappa_q
        CS%Txy(I,J,k) = CS%kappa_q(I,J) * (CS%vort_xy(I,J,k) * sh_xx_q)

      enddo ; enddo
    endif

  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_stress)

end subroutine compute_stress

! Interpolation on a unit square is explained here:
! https://en.wikipedia.org/wiki/Bilinear_interpolation#:~:text=On%20the%20unit-,square,-%5Bedit%5D
pure function bilin_interp(f00, f01, f10, f11, x, y) result(interpolated)
  real, intent(in) :: f00, f01, f10, f11, x, y
  real :: interpolated

  interpolated = &
    f00 * (1-x) * (1-y) + &
    f01 * (1-x) * y     + &
    f10 * x * (1-y)     + &
    f11 * x * y

end function bilin_interp

!> Compute local energy source due to ZB model
!! and Smagorinsky (Laplacian) model and average in lagrangian frame
!! Then, Smagorinsky model is applied to enforce KE conservation
!! on lagrangian particles
subroutine Bound_backscatter_lagrangian(u, v, h, G, GV, CS)
  type(ocean_grid_type),   intent(in)    :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
        intent(in)    :: u  !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
        intent(in)    :: v  !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
        intent(in) :: h     !< Layer thicknesses [H ~> m or kg m-2].

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n
  integer :: i0, j0 ! Index of the left bottom corner of the unit square used for interpolation

  real, dimension(SZI_(G),SZJ_(G))            :: shear       ! Shear of Smagorinsky model [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV))   :: Txx         ! Smagorinsky model
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)) :: Txy         ! Smagorinsky model
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV))   :: Smag_coef   ! Local Smagorinsky coefficient
  
  real, dimension(SZI_(G),SZJ_(G)) :: Esource_smag  ! Source of Smagorinsky model
  real, dimension(SZI_(G),SZJ_(G)) :: Esource_ZB    ! Source of ZB model

  real :: Tdd ! Deviatoric stress of ZB model
  real :: uT, vT ! Interpolation of advective velocity to T points
  real :: dx, dy ! Displacement of the fluid particle backward in time in metres
  real :: x, y   ! Nondimension coordinate on unit square of the interpolation point
  real :: Smag_coef_mean ! Volume-averaged Smagorinsky coefficient

  call cpu_clock_begin(CS%id_clock_cdiss)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  shear = 0.
  Txx = 0.
  Txy = 0.
  Esource_smag = 0.
  Esource_ZB = 0.

  do k=1,nz

    do j=js-1,je+1 ; do i=is-1,ie+1
      shear(i,j) = sqrt(CS%sh_xx(i,j,k)**2 + 0.25 * (     &
        (CS%sh_xy(I-1,J-1,k)**2 + CS%sh_xy(I,J  ,k)**2)   &
      + (CS%sh_xy(I-1,J  ,k)**2 + CS%sh_xy(I,J-1,k)**2)   &
      ))
    enddo; enddo

    do j=js-1,je+1 ; do i=is-1,ie+1
      Txx(i,j,k) = G%areaT(i,j) * G%mask2dT(i,j) * shear(i,j) * CS%sh_xx(i,j,k)
    enddo; enddo

    do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
      Txy(I,J,k) = G%areaBu(I,J) * G%mask2dBu(I,J) * & 
        0.25 * ((shear(i,j) + shear(i+1,j+1)) + (shear(i+1,j) + shear(i,j+1))) * &
        CS%sh_xy(I,J,k)
    enddo; enddo

    do j=js-1,je+1 ; do i=is-1,ie+1
      ! Note here we account for deviatoric stress and for off-diagonal term once
      ! Probably, more accurate source term should be multiplied by two
      ! Note that it is dissipation. I.e., if model dissipates, like Smagorinsky, it is positive
      Esource_smag(i,j) = h(i,j,k) * Txx(i,j,k) * CS%sh_xx(i,j,k) +                          &
        0.25 * G%IareaT(i,j) *                                                               &
        (                                                                                    &
          (G%areaBu(I,J) * CS%hq(I,J,k) * Txy(I,J,k) * CS%sh_xy(I,J,k) +                     &
           G%areaBu(I-1,J-1) * CS%hq(I-1,J-1,k) * Txy(I-1,J-1,k) * CS%sh_xy(I-1,J-1,k)) +    &
          (G%areaBu(I-1,J) * CS%hq(I-1,J,k) * Txy(I-1,J,k) * CS%sh_xy(I-1,J,k) +             &
           G%areaBu(I,J-1) * CS%hq(I,J-1,k) * Txy(I,J-1,k) * CS%sh_xy(I,J-1,k))              &
        )
      
      ! Here we neglect contribution of the trace part which correlated only with divergence
      Tdd = 0.5 * (CS%Txx(i,j,k) - CS%Tyy(i,j,k))
      Esource_ZB(i,j) = h(i,j,k) * Tdd * CS%sh_xx(i,j,k) +                                   &
        0.25 * G%IareaT(i,j) *                                                               &
        (                                                                                    &
          (G%areaBu(I,J) * CS%hq(I,J,k) * CS%Txy(I,J,k) * CS%sh_xy(I,J,k) +                  &
           G%areaBu(I-1,J-1) * CS%hq(I-1,J-1,k) * CS%Txy(I-1,J-1,k) * CS%sh_xy(I-1,J-1,k)) + &
          (G%areaBu(I-1,J) * CS%hq(I-1,J,k) * CS%Txy(I-1,J,k) * CS%sh_xy(I-1,J,k) +          &
           G%areaBu(I,J-1) * CS%hq(I,J-1,k) * CS%Txy(I,J-1,k) * CS%sh_xy(I,J-1,k))           &
        )

      ! Transform energy source term to the relaxation term of lagrangially-averaged equation
      Esource_smag(i,j) = CS%dt * shear(i,j) * (Esource_smag(i,j) - CS%Esource_smag(i,j,k))
      Esource_ZB(i,j)   = CS%dt * shear(i,j) * (Esource_ZB(i,j)   - CS%Esource_ZB(i,j,k))

      ! Below we update the relaxation term with value on a fluid particle backward in time
      uT = (u(I,j,k) + u(I-1,j,k)) * 0.5
      vT = (v(i,J,k) + v(i,J-1,k)) * 0.5
      
      ! Displacement backward in time, so the sign is minus
      dx = - uT * CS%dt
      dy = - vT * CS%dt

      ! Now we determine the left bottom corner of the unit square
      ! used for interpolation.
      if (dx > 0) then
        i0 = i
      else
        i0 = i-1
      endif

      if (dy>0) then
        j0 = j
      else
        j0 = j-1
      endif

      ! Here we determine the relative position of the interpolation point
      ! Note in every case the corner point is the center of the square
      ! on which bilinear interpolation is performed.
      ! If dx<0, we need to convert the non-dimensional coordinate to positive one,
      ! which is w.r.t. leftmost corner of the unit square
      if (dx > 0.) then
        x = dx / G%dxBu(i0,j0)
      else
        x = 1. + dx / G%dxBu(i0,j0) ! this is smaller than 1
      endif
      
      if (dy > 0.) then
        y = dy / G%dyBu(i0,j0)
      else
        y = 1. + dy / G%dyBu(i0,j0)
      endif
      
      ! we update the relaxation term with a value on a fluid particle backward in time
      Esource_smag(i,j) = Esource_smag(i,j) +                                          &
          bilin_interp(CS%Esource_smag(i0,j0,k),   CS%Esource_smag(i0,j0+1,k),         &
                       CS%Esource_smag(i0+1,j0,k), CS%Esource_smag(i0+1,j0+1,k), x, y)

      Esource_ZB(i,j) = Esource_ZB(i,j) +                                          &
          bilin_interp(CS%Esource_ZB(i0,j0,k),   CS%Esource_ZB(i0,j0+1,k),         &
                       CS%Esource_ZB(i0+1,j0,k), CS%Esource_ZB(i0+1,j0+1,k), x, y)
    enddo; enddo

    ! Save computations to storage array and enforcing zero B.C.
    CS%Esource_smag(:,:,k) = Esource_smag(:,:) * G%mask2dT(:,:)
    CS%Esource_ZB(:,:,k) = Esource_ZB(:,:) * G%mask2dT(:,:)
    
  enddo ! end of k loop

  Smag_coef = 0.
  ! Remove excessive energy backscatter, if it is present
  do k=1,nz
    do j=js-1,je+1 ; do i=is-1,ie+1
      ! If Smagorinsky locally dissipates energy and ANN locally generates energy (dissipatioin is negative)
      if (CS%Esource_smag(i,j,k) > 0. .and. CS%Esource_ZB(i,j,k) < 0.) then
        Smag_coef(i,j,k) = - CS%Esource_ZB(i,j,k) / CS%Esource_smag(i,j,k)
      endif
    enddo; enddo
    ! Update the Smagorinsky model with computed coefficient
    Txx(:,:,k) = Txx(:,:,k) * Smag_coef(:,:,k)
    do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
      Txy(I,J,k) = Txy(I,J,k) * &
        0.25 * ((Smag_coef(i,j,k) + Smag_coef(i+1,j+1,k)) + (Smag_coef(i+1,j,k) + Smag_coef(i,j+1,k)))
    enddo; enddo
  enddo

  CS%Txx = CS%Txx + Txx
  CS%Txy = CS%Txy + Txy
  CS%Tyy = CS%Tyy - Txx ! Because we are working with deviatoric stress

  if (CS%id_Txx_smag)  call post_data(CS%id_Txx_smag, Txx, CS%diag)
  if (CS%id_Txy_smag)  call post_data(CS%id_Txy_smag, Txy, CS%diag)
  if (CS%id_Esrc_smag) call post_data(CS%id_Esrc_smag, CS%Esource_smag, CS%diag)
  if (CS%id_Esrc_ZB)   call post_data(CS%id_Esrc_ZB, CS%Esource_ZB, CS%diag)
  if (CS%id_smag_coef) call post_data(CS%id_smag_coef, Smag_coef, CS%diag)
  if (CS%id_smag) then
    Smag_coef_mean = global_volume_mean(Smag_coef, h, G, GV)
    call post_data(CS%id_smag, Smag_coef_mean, CS%diag)
  endif

  call cpu_clock_end(CS%id_clock_cdiss)

end subroutine Bound_backscatter_lagrangian

pure function norm(x,n) result (y)
  real, dimension(n), intent(in) :: x
  integer, intent(in) :: n
  real :: y

  integer :: i

  y = 0.
  do i=1,n
    y = y + x(i)**2
  enddo
  y = sqrt(y)
end function norm

!> Compute stress tensor T =
!! (Txx, Txy;
!!  Txy, Tyy)
!!  with ANN
subroutine compute_stress_ANN(G, GV, CS)
  type(ocean_grid_type),   intent(in)    :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  real :: x(12), y(3)
  real :: input_norm

  call cpu_clock_begin(CS%id_clock_stress_ANN)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  call pass_var(CS%sh_xy, G%Domain, clock=CS%id_clock_mpi, position=CORNER)
  call pass_var(CS%sh_xx, G%Domain, clock=CS%id_clock_mpi)
  call pass_var(CS%vort_xy, G%Domain, clock=CS%id_clock_mpi, position=CORNER)

  do k=1,nz
    do j=js-2,je+2 ; do i=is-2,ie+2
      x(1) = CS%sh_xy(i-1,j-1,k)
      x(2) = CS%sh_xy(i,j-1,k)
      x(3) = CS%sh_xy(i-1,j,k)
      x(4) = CS%sh_xy(i,j,k)

      x(5) = CS%sh_xx(i,j,k)
      x(6) = CS%sh_xx(i+1,j,k)
      x(7) = CS%sh_xx(i,j+1,k)
      x(8) = CS%sh_xx(i+1,j+1,k)

      x(9)  = CS%vort_xy(i-1,j-1,k)
      x(10) = CS%vort_xy(i,j-1,k)
      x(11) = CS%vort_xy(i-1,j,k)
      x(12) = CS%vort_xy(i,j,k)

      input_norm = norm(x,12)
      x = x / (input_norm + CS%subroundoff_shear)

      call ANN_apply(x, y, CS%ann_instance)

      y = y * input_norm * input_norm
   
      CS%Txx(i,j,k) = CS%kappa_h(i,j) * y(1)
      CS%Tyy(i,j,k) = CS%kappa_h(i,j) * y(2)
      CS%Txy(I,J,k) = CS%kappa_q(I,J) * y(3)
    enddo ; enddo
  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_stress_ANN)

end subroutine compute_stress_ANN

!> Compute stress tensor T =
!! (Txx, Txy;
!!  Txy, Tyy)
!!  with ANN
subroutine compute_stress_ANN_3x3(G, GV, CS)
  type(ocean_grid_type),   intent(in)    :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  real :: x(27), y1(1), y2(2), y3(1), y4(2)
  real :: input_norm
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)) :: &
        sh_xx_q,   & ! sh_xx interpolated to the corner [T-1 ~ s-1]
        norm_q       ! Norm in q points [T-1 ~ s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: &
        sh_xy_h,   & ! sh_xy interpolated to the center [T-1 ~ s-1]
        vort_xy_h, & ! vort_xy interpolated to the center [T-1 ~ s-1]
        norm_h       ! Norm in h points [T-1 ~ s-1]

  real, dimension(SZI_(G),SZJ_(G)) :: &
        sqr_h ! Sum of squares in h points
  real, dimension(SZIB_(G),SZJB_(G)) :: &
        sqr_q ! Sum of squares in q points

  real :: diagonal, side, Ttr, Tdd

  call cpu_clock_begin(CS%id_clock_stress_ANN)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  sh_xx_q = 0.
  sh_xy_h = 0.
  vort_xy_h = 0.
  norm_q = 0.
  norm_h = 0.

  call pass_var(CS%sh_xy, G%Domain, clock=CS%id_clock_mpi, position=CORNER)
  call pass_var(CS%sh_xx, G%Domain, clock=CS%id_clock_mpi)
  call pass_var(CS%vort_xy, G%Domain, clock=CS%id_clock_mpi, position=CORNER)

  ! Interpolate input features
  do k=1,nz
    do j=js-1,je+1 ; do i=is-1,ie+1
      ! It is assumed that B.C. is applied to sh_xy and vort_xy
      sh_xy_h(i,j,k) = 0.25 * ( (CS%sh_xy(I-1,J-1,k) + CS%sh_xy(I,J,k)) &
                       + (CS%sh_xy(I-1,J,k) + CS%sh_xy(I,J-1,k)) ) * G%mask2dT(i,j)

      vort_xy_h(i,j,k) = 0.25 * ( (CS%vort_xy(I-1,J-1,k) + CS%vort_xy(I,J,k)) &
                         + (CS%vort_xy(I-1,J,k) + CS%vort_xy(I,J-1,k)) ) * G%mask2dT(i,j)
      
      sqr_h(i,j) = CS%sh_xx(i,j,k)**2 + sh_xy_h(i,j,k)**2 + vort_xy_h(i,j,k)**2
    enddo; enddo

    do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
      sh_xx_q(I,J,k) = 0.25 * ( (CS%sh_xx(i+1,j+1,k) + CS%sh_xx(i,j,k)) &
                       + (CS%sh_xx(i+1,j,k) + CS%sh_xx(i,j+1,k))) * G%mask2dBu(I,J)
      sqr_q(I,J) = sh_xx_q(I,J,k)**2 + CS%vort_xy(I,J,k)**2 + CS%sh_xy(I,J,k)**2
    enddo; enddo

    do j=js,je ; do i=is,ie
      diagonal = (sqr_h(i+1,j+1) + sqr_h(i-1,j-1)) + (sqr_h(i+1,j-1) + sqr_h(i-1,j+1))
      side     = (sqr_h(i+1,j) + sqr_h(i-1,j)) + (sqr_h(i,j+1) + sqr_h(i,j-1))
      norm_h(i,j,k) = sqrt((diagonal + side) + sqr_h(i,j))
    enddo; enddo

    do J=Jsq,Jeq ; do I=Isq,Ieq
      diagonal = (sqr_q(i+1,j+1) + sqr_q(i-1,j-1)) + (sqr_q(i+1,j-1) + sqr_q(i-1,j+1))
      side     = (sqr_q(i+1,j) + sqr_q(i-1,j)) + (sqr_q(i,j+1) + sqr_q(i,j-1))
      norm_q(i,j,k) = sqrt((diagonal + side) + sqr_q(i,j))
    enddo; enddo 
  enddo

  call pass_var(sh_xy_h, G%Domain, clock=CS%id_clock_mpi)
  call pass_var(vort_xy_h, G%Domain, clock=CS%id_clock_mpi)
  call pass_var(sh_xx_q, G%Domain, clock=CS%id_clock_mpi, position=CORNER)
  call pass_var(norm_h, G%Domain, clock=CS%id_clock_mpi) 
  call pass_var(norm_q, G%Domain, clock=CS%id_clock_mpi, position=CORNER)

  do k=1,nz
    ! compute Txx, Tyy tensor
    do j=js-1,je+1 ; do i=is-1,ie+1
      x(1:9) = RESHAPE(sh_xy_h(i-1:i+1,j-1:j+1,k), (/9/))
      x(10:18) = RESHAPE(CS%sh_xx(i-1:i+1,j-1:j+1,k), (/9/))
      x(19:27) = RESHAPE(vort_xy_h(i-1:i+1,j-1:j+1,k), (/9/))

      input_norm = norm_h(i,j,k)

      x = x / (input_norm + CS%subroundoff_shear)

      call ANN_apply(x, y2, CS%ann_Txx_Tyy)

      y2 = y2 * input_norm * input_norm

      if (CS%rotation_invariant) then
        x(1:9) = RESHAPE(TRANSPOSE(-sh_xy_h(i-1:i+1,j-1:j+1,k)), (/9/))
        x(10:18) = RESHAPE(TRANSPOSE(-CS%sh_xx(i-1:i+1,j-1:j+1,k)), (/9/))
        x(19:27) = RESHAPE(TRANSPOSE(vort_xy_h(i-1:i+1,j-1:j+1,k)), (/9/))

        input_norm = norm_h(i,j,k)

        x = x / (input_norm + CS%subroundoff_shear)

        call ANN_apply(x, y4, CS%ann_Txx_Tyy)

        y4 = y4 * input_norm * input_norm

        Ttr = (y2(1) + y2(2)) * 0.25 + (y4(1) + y4(2)) * 0.25
        Tdd = (y2(1) - y2(2)) * 0.25 - (y4(1) - y4(2)) * 0.25
      else
        Ttr = (y2(1) + y2(2)) * 0.5
        Tdd = (y2(1) - y2(2)) * 0.5
      endif

      CS%Txx(i,j,k) = CS%kappa_h(i,j) * (Ttr + Tdd)
      CS%Tyy(i,j,k) = CS%kappa_h(i,j) * (Ttr - Tdd)
    enddo ; enddo

    ! compute Txy
    do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
      x(1:9) = RESHAPE(CS%sh_xy(i-1:i+1,j-1:j+1,k), (/9/))
      x(10:18) = RESHAPE(sh_xx_q(i-1:i+1,j-1:j+1,k), (/9/))
      x(19:27) = RESHAPE(CS%vort_xy(i-1:i+1,j-1:j+1,k), (/9/))

      input_norm = norm_q(I,J,k)
      x = x / (input_norm + CS%subroundoff_shear)

      call ANN_apply(x, y1, CS%ann_Txy)

      y1 = y1 * input_norm * input_norm

      if (CS%rotation_invariant) then
        x(1:9) = RESHAPE(TRANSPOSE(-CS%sh_xy(i-1:i+1,j-1:j+1,k)), (/9/))
        x(10:18) = RESHAPE(TRANSPOSE(-sh_xx_q(i-1:i+1,j-1:j+1,k)), (/9/))
        x(19:27) = RESHAPE(TRANSPOSE(CS%vort_xy(i-1:i+1,j-1:j+1,k)), (/9/))

        input_norm = norm_q(I,J,k)
        x = x / (input_norm + CS%subroundoff_shear)

        call ANN_apply(x, y3, CS%ann_Txy)

        y3 = y3 * input_norm * input_norm

        CS%Txy(I,J,k) = CS%kappa_q(I,J) * (y1(1) - y3(1)) * 0.5
      else
        CS%Txy(I,J,k) = CS%kappa_q(I,J) * y1(1)
      endif

    enddo; enddo
  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_stress_ANN)

end subroutine compute_stress_ANN_3x3

!> Compute the divergence of subgrid stress
!! weighted with thickness, i.e.
!! (fx,fy) = 1/h Div(h * [Txx, Txy; Txy, Tyy])
!! and update the acceleration due to eddy viscosity as
!! diffu = diffu + dx; diffv = diffv + dy
!! Optionally, before computing the divergence, we attenuate the stress
!! according to the Klower formula.
!! In symmetric memory model: Txx, Tyy, Txy, c_diss should have halo 1
!! with applied zero B.C.
subroutine compute_stress_divergence(u, v, h, diffu, diffv, dx2h, dy2h, dx2q, dy2q, G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS   !< ZB2020 control structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
        intent(in)    :: u  !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
        intent(in)    :: v  !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
        intent(in) :: h             !< Layer thicknesses [H ~> m or kg m-2].
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
        intent(out) :: diffu           !< Zonal acceleration due to convergence of
                                       !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
        intent(out) :: diffv           !< Meridional acceleration due to convergence
                                       !! of along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJ_(G)),           &
        intent(in) :: dx2h          !< dx^2 at h points [L2 ~> m2]
  real, dimension(SZI_(G),SZJ_(G)),           &
        intent(in) :: dy2h          !< dy^2 at h points [L2 ~> m2]
  real, dimension(SZIB_(G),SZJB_(G)),         &
        intent(in) :: dx2q          !< dx^2 at q points [L2 ~> m2]
  real, dimension(SZIB_(G),SZJB_(G)),         &
        intent(in) :: dy2q          !< dy^2 at q points [L2 ~> m2]

  ! Local variables
  real, dimension(SZI_(G),SZJ_(G)) :: &
        Mxx, & ! Subgrid stress Txx multiplied by thickness and dy^2 [H L4 T-2 ~> m5 s-2]
        Myy    ! Subgrid stress Tyy multiplied by thickness and dx^2 [H L4 T-2 ~> m5 s-2]

  real, dimension(SZIB_(G),SZJB_(G)) :: &
        Mxy    ! Subgrid stress Txy multiplied by thickness [H L2 T-2 ~> m3 s-2]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: &
        Txx_smooth, & ! Subgrid stress Txx multiplied by thickness and dy^2 [H L4 T-2 ~> m5 s-2]
        Tyy_smooth    ! Subgrid stress Tyy multiplied by thickness and dx^2 [H L4 T-2 ~> m5 s-2]

  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)) :: &
        Txy_smooth    ! Subgrid stress Txy multiplied by thickness [H L2 T-2 ~> m3 s-2]

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)) :: &
        ZB2020u, &        !< Zonal acceleration due to convergence of
                          !! along-coordinate stress tensor for ZB model
                          !! [L T-2 ~> m s-2]
        ZB2020u_smooth
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)) :: &
        ZB2020v, &        !< Meridional acceleration due to convergence
                          !! of along-coordinate stress tensor for ZB model
                          !! [L T-2 ~> m s-2]
        ZB2020v_smooth

  real :: KE_term(SZI_(G),SZJ_(G),SZK_(GV)) !< A term in the kinetic energy budget
                                            ! [H L2 T-3 ~> m3 s-3 or W m-2]

  real :: h_u ! Thickness interpolated to u points [H ~> m or kg m-2].
  real :: h_v ! Thickness interpolated to v points [H ~> m or kg m-2].
  real :: fx  ! Zonal acceleration      [L T-2 ~> m s-2]
  real :: fy  ! Meridional acceleration [L T-2 ~> m s-2]

  real :: h_neglect    ! Thickness so small it can be lost in
                       ! roundoff and so neglected [H ~> m or kg m-2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k
  logical :: save_ZB2020u, save_ZB2020v ! Save the acceleration due to ZB2020 model
  integer :: remaining_iterations, current_halo

  real :: global_integral_smag, global_integral_ZB2020
  real :: global_integral_smooth, global_integral_residual
  real :: SMAG_BI_CONST
  real :: attenuation, current_backscatter_ratio
  character(len=100) :: message

  call cpu_clock_begin(CS%id_clock_divergence)

  if (CS%backscatter_ratio > 0.) then
    Txx_smooth = CS%Txx
    Tyy_smooth = CS%Tyy
    Txy_smooth = CS%Txy

    call pass_var(Txy_smooth, G%Domain, clock=CS%id_clock_mpi, position=CORNER)
    call pass_var(Txx_smooth, G%Domain, clock=CS%id_clock_mpi)
    call pass_var(Tyy_smooth, G%Domain, clock=CS%id_clock_mpi)

    current_halo=4; remaining_iterations = 1
    call filter_hq(G, GV, CS, current_halo, remaining_iterations, q=Txy_smooth)
    current_halo=4; remaining_iterations = 1
    call filter_hq(G, GV, CS, current_halo, remaining_iterations, h=Txx_smooth)
    current_halo=4; remaining_iterations = 1
    call filter_hq(G, GV, CS, current_halo, remaining_iterations, h=Tyy_smooth)
  endif

  save_ZB2020u = (CS%id_ZB2020u > 0) .or. (CS%id_KE_ZB2020 > 0)
  save_ZB2020v = (CS%id_ZB2020v > 0) .or. (CS%id_KE_ZB2020 > 0)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  h_neglect  = GV%H_subroundoff

  ZB2020u = 0.
  ZB2020v = 0.

  do k=1,nz
    if (CS%Klower_R_diss > 0) then
      do J=js-1,Jeq ; do I=is-1,Ieq
          Mxy(I,J) = (CS%Txy(I,J,k) *                                         &
                      (0.25 * ( (CS%c_diss(i,j  ,k) + CS%c_diss(i+1,j+1,k))   &
                              + (CS%c_diss(i,j+1,k) + CS%c_diss(i+1,j  ,k)))  &
                      )                                                       &
                     ) * CS%hq(I,J,k)
          if (CS%backscatter_ratio>0.) then
            Txy_smooth(I,J,k) = (Txy_smooth(I,J,k) *                                    &
                                (0.25 * ( (CS%c_diss(i,j  ,k) + CS%c_diss(i+1,j+1,k))   &
                                        + (CS%c_diss(i,j+1,k) + CS%c_diss(i+1,j  ,k)))  &
                                )                                                       &
                              ) * CS%hq(I,J,k)
          endif
      enddo ; enddo
    else
      do J=js-1,Jeq ; do I=is-1,Ieq
        Mxy(I,J) = CS%Txy(I,J,k) * CS%hq(I,J,k)
        if (CS%backscatter_ratio>0.) then
          Txy_smooth(I,J,k) = Txy_smooth(I,J,k) * CS%hq(I,J,k)
        endif
      enddo ; enddo
    endif

    if (CS%Klower_R_diss > 0) then
      do j=js-1,je+1 ; do i=is-1,ie+1
        Mxx(i,j) = ((CS%Txx(i,j,k) * CS%c_diss(i,j,k)) * h(i,j,k)) * dy2h(i,j)
        Myy(i,j) = ((CS%Tyy(i,j,k) * CS%c_diss(i,j,k)) * h(i,j,k)) * dx2h(i,j)
        if (CS%backscatter_ratio>0.) then
          Txx_smooth(i,j,k) = ((Txx_smooth(i,j,k) * CS%c_diss(i,j,k)) * h(i,j,k)) * dy2h(i,j)
          Tyy_smooth(i,j,k) = ((Tyy_smooth(i,j,k) * CS%c_diss(i,j,k)) * h(i,j,k)) * dx2h(i,j)
        endif
      enddo ; enddo
    else
      do j=js-1,je+1 ; do i=is-1,ie+1
        Mxx(i,j) = ((CS%Txx(i,j,k)) * h(i,j,k)) * dy2h(i,j)
        Myy(i,j) = ((CS%Tyy(i,j,k)) * h(i,j,k)) * dx2h(i,j)
        if (CS%backscatter_ratio>0.) then
          Txx_smooth(i,j,k) = (Txx_smooth(i,j,k) * h(i,j,k)) * dy2h(i,j)
          Tyy_smooth(i,j,k) = (Tyy_smooth(i,j,k) * h(i,j,k)) * dx2h(i,j)
        endif
      enddo ; enddo
    endif

    ! Evaluate 1/h x.Div(h S) (Line 1495 of MOM_hor_visc.F90)
    ! Minus occurs because in original file (du/dt) = - div(S),
    ! but here is the discretization of div(S)
    do j=js,je ; do I=Isq,Ieq
      h_u = 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i+1,j)*h(i+1,j,k)) + h_neglect
      fx = -((G%IdyCu(I,j)*(Mxx(i,j)                - &
                            Mxx(i+1,j))             + &
              G%IdxCu(I,j)*(dx2q(I,J-1)*Mxy(I,J-1)  - &
                            dx2q(I,J)  *Mxy(I,J)))  * &
              G%IareaCu(I,j)) / h_u
      if (not(CS%ann_smag_conserv) .and. not(CS%backscatter_ratio>0.)) &
        diffu(I,j,k) = diffu(I,j,k) + fx
      if (save_ZB2020u .or. CS%ann_smag_conserv .or. GM_conserv .or. CS%backscatter_ratio > 0.) &
        ZB2020u(I,j,k) = fx
      if (CS%backscatter_ratio > 0.) then
        ZB2020u_smooth(I,j,k) =                                &
             -((G%IdyCu(I,j)*(Txx_smooth(i,j,k)              - &
                              Txx_smooth(i+1,j,k))           + &
              G%IdxCu(I,j)*(dx2q(I,J-1)*Txy_smooth(I,J-1,k)  - &
                            dx2q(I,J)  *Txy_smooth(I,J,k)))  * &
              G%IareaCu(I,j)) / h_u
      endif
    enddo ; enddo

    ! Evaluate 1/h y.Div(h S) (Line 1517 of MOM_hor_visc.F90)
    do J=Jsq,Jeq ; do i=is,ie
      h_v = 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i,j+1)*h(i,j+1,k)) + h_neglect
      fy = -((G%IdyCv(i,J)*(dy2q(I-1,J)*Mxy(I-1,J)  - &
                            dy2q(I,J)  *Mxy(I,J))   + & ! NOTE this plus
              G%IdxCv(i,J)*(Myy(i,j)                - &
                            Myy(i,j+1)))            * &
              G%IareaCv(i,J)) / h_v
      if (not(CS%ann_smag_conserv) .and. not(CS%backscatter_ratio>0.)) &
        diffv(i,J,k) = diffv(i,J,k) + fy
      if (save_ZB2020v .or. CS%ann_smag_conserv .or. GM_conserv .or. CS%backscatter_ratio > 0.) &
        ZB2020v(i,J,k) = fy
      if (CS%backscatter_ratio > 0.) then
        ZB2020v_smooth(i,J,k) =                                  &
             -((G%IdyCv(i,J)*(dy2q(I-1,J)*Txy_smooth(I-1,J,k)  - &
                              dy2q(I,J)  *Txy_smooth(I,J,k))   + & ! NOTE this plus
              G%IdxCv(i,J)*(Tyy_smooth(i,j,k)                  - &
                            Tyy_smooth(i,j+1,k)))              * &
              G%IareaCv(i,J)) / h_v
      endif
    enddo ; enddo

  enddo ! end of k loop

  ! Here we choose the Smagorinsky coefficient such that the total KE energy contribution from
  ! ANN + biharmonic Smagorinsky model is zero
  if (CS%ann_smag_conserv) then
    call compute_energy_source(u, v, h, diffu, diffv, G, GV, CS, KE_term, global_integral = global_integral_smag)
    call compute_energy_source(u, v, h, ZB2020u, ZB2020v, G, GV, CS, KE_term, global_integral = global_integral_ZB2020)
    SMAG_BI_CONST = 0.
    if (global_integral_ZB2020 > 0 .and. global_integral_smag < 0) then
      SMAG_BI_CONST = - global_integral_ZB2020 / global_integral_smag
    endif
    diffu = diffu * SMAG_BI_CONST
    diffv = diffv * SMAG_BI_CONST
    diffu = diffu + ZB2020u
    diffv = diffv + ZB2020v
    global_integral_smag = global_integral_smag * SMAG_BI_CONST

    if (CS%id_smag > 0) call post_data(CS%id_smag, SMAG_BI_CONST, CS%diag)
    if (CS%id_KE_smag > 0) call post_data(CS%id_KE_smag, global_integral_smag, CS%diag)

    ! write(message, '(a, g12.6)') 'Global Smagorinsky KE contribution: ', global_integral_smag
    ! call MOM_mesg(message, 2)
    ! write(message, '(a, g12.6)') 'Global ZB2020 KE contribution: ', global_integral_ZB2020
    ! call MOM_mesg(message, 2)
    ! write(message, '(a, g12.6)') 'Smagorinsky coefficient: ', SMAG_BI_CONST
    ! call MOM_mesg(message, 2)
  endif

  ! Here we choose the GM diffusivity coefficient such that the total KE energy contribution from
  ! ANN + GM model is zero
  if (GM_conserv) then
    call compute_energy_source(u, v, h, ZB2020u, ZB2020v, G, GV, CS, KE_term, global_integral = global_integral_ZB2020)
    GM_coefficient = 1e-10
    if (global_integral_ZB2020 > 0. .and. GM_dissipation < 0.) then
      GM_coefficient = - global_integral_ZB2020 / GM_dissipation
    endif

    if (CS%id_GM_coef>0) call post_data(CS%id_GM_coef, GM_coefficient, CS%diag)
    if (CS%id_PE_GM>0) call post_data(CS%id_PE_GM, GM_dissipation * GM_coefficient, CS%diag)

  endif

  if (CS%backscatter_ratio > 0.) then
    call compute_energy_source(u, v, h, ZB2020u, ZB2020v, G, GV, CS, KE_term, global_integral = global_integral_ZB2020)
    call compute_energy_source(u, v, h, ZB2020u_smooth, ZB2020v_smooth, G, GV, CS, KE_term, global_integral = global_integral_smooth)
    global_integral_residual = global_integral_ZB2020 - global_integral_smooth
    attenuation = 1.0
    ! Backscatter ratio can be defined only if energetic contributions have different signs
    if (global_integral_residual < 0. .and. global_integral_smooth > 0.) then
      current_backscatter_ratio = - global_integral_smooth / global_integral_residual
      if (current_backscatter_ratio > CS%backscatter_ratio) then
        attenuation = CS%backscatter_ratio / current_backscatter_ratio
        ! Later I actually allowed for ratio more than 1
        ! if (attenuation > 1.) then
        !   write(*,*) 'Warning: attenuation > 1.'
        ! endif
        if (attenuation < 0.) then
          write(*,*) 'Warning: attenuation < 0.'
        endif
        ! The final model will be ZB_residual + attenuation * ZB_smooth
        ! Which is equivalent to (ZB - ZB_smooth) + attenuation * ZB_smooth = 
        ! ZB + (attenuation-1) * smooth
        ZB2020u = ZB2020u + (attenuation-1) * ZB2020u_smooth
        ZB2020v = ZB2020v + (attenuation-1) * ZB2020v_smooth
      endif
    endif
    diffu = diffu + ZB2020u
    diffv = diffv + ZB2020v
    if (CS%id_attenuation) call post_data(CS%id_attenuation, attenuation, CS%diag)
  endif

  call cpu_clock_end(CS%id_clock_divergence)

  call cpu_clock_begin(CS%id_clock_post)
  if (CS%id_ZB2020u>0)   call post_data(CS%id_ZB2020u, ZB2020u, CS%diag)
  if (CS%id_ZB2020v>0)   call post_data(CS%id_ZB2020v, ZB2020v, CS%diag)
  call cpu_clock_end(CS%id_clock_post)

  call cpu_clock_begin(CS%id_clock_post)
  call compute_energy_source(u, v, h, ZB2020u, ZB2020v, G, GV, CS, KE_term, global_integral = global_integral_ZB2020)
  if (CS%id_KE_ZB2020>0) call post_data(CS%id_KE_ZB2020, KE_term, CS%diag)
  if (CS%id_KE_ZB > 0) call post_data(CS%id_KE_ZB, global_integral_ZB2020, CS%diag)
  call cpu_clock_end(CS%id_clock_post)

end subroutine compute_stress_divergence

!> Filtering of the velocity gradients sh_xx, sh_xy, vort_xy.
!! Here instead of smoothing we do sharpening, i.e.
!! return (initial - smoothed) fields.
!! The algorithm: marching halo with non-blocking grouped MPI
!! exchanges. The input array sh_xx should have halo 2 with
!! applied zero B.C. The arrays sh_xy and vort_xy should have
!! halo 1 with applied B.C. The output have the same halo and B.C.
subroutine filter_velocity_gradients(G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  real, dimension(SZI_(G), SZJ_(G), SZK_(GV)) :: &
        sh_xx          ! Copy of CS%sh_xx [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)) :: &
        sh_xy, vort_xy ! Copy of CS%sh_xy and CS%vort_xy [T-1 ~> s-1]

  integer :: xx_halo, xy_halo, vort_halo ! currently available halo for gradient components
  integer :: xx_iter, xy_iter, vort_iter ! remaining number of iterations
  integer :: niter                       ! required number of iterations

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  niter = CS%HPF_iter

  if (niter == 0) return

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  if (.not. G%symmetric) &
    call do_group_pass(CS%pass_xx, G%Domain, &
      clock=CS%id_clock_mpi)

  ! This is just copy of the array
  call cpu_clock_begin(CS%id_clock_filter)
  do k=1,nz
    ! Halo of size 2 is valid
    do j=js-2,je+2; do i=is-2,ie+2
      sh_xx(i,j,k) = CS%sh_xx(i,j,k)
    enddo; enddo
    ! Only halo of size 1 is valid
    do J=Jsq-1,Jeq+1; do I=Isq-1,Ieq+1
      sh_xy(I,J,k) = CS%sh_xy(I,J,k)
      vort_xy(I,J,k) = CS%vort_xy(I,J,k)
    enddo; enddo
  enddo
  call cpu_clock_end(CS%id_clock_filter)

  xx_halo = 2; xy_halo = 1; vort_halo = 1;
  xx_iter = niter; xy_iter = niter; vort_iter = niter;

  do while &
    (xx_iter >  0 .or. xy_iter >  0 .or. & ! filter iterations remain to be done
    xx_halo < 2 .or. xy_halo < 1)        ! there is no halo for VG tensor

    ! ---------- filtering sh_xx ---------
    if (xx_halo < 2) then
      call complete_group_pass(CS%pass_xx, G%Domain, clock=CS%id_clock_mpi)
      xx_halo = CS%HPF_halo
    endif

    call filter_hq(G, GV, CS, xx_halo, xx_iter, h=CS%sh_xx)

    if (xx_halo < 2) &
      call start_group_pass(CS%pass_xx, G%Domain, clock=CS%id_clock_mpi)

    ! ------ filtering sh_xy, vort_xy ----
    if (xy_halo < 1) then
      call complete_group_pass(CS%pass_xy, G%Domain, clock=CS%id_clock_mpi)
      xy_halo = CS%HPF_halo; vort_halo = CS%HPF_halo
    endif

    call filter_hq(G, GV, CS, xy_halo, xy_iter, q=CS%sh_xy)
    call filter_hq(G, GV, CS, vort_halo, vort_iter, q=CS%vort_xy)

    if (xy_halo < 1) &
      call start_group_pass(CS%pass_xy, G%Domain, clock=CS%id_clock_mpi)

  enddo

  ! We implement sharpening by computing residual
  ! B.C. are already applied to all fields
  call cpu_clock_begin(CS%id_clock_filter)
  do k=1,nz
    do j=js-2,je+2; do i=is-2,ie+2
      CS%sh_xx(i,j,k) = sh_xx(i,j,k) - CS%sh_xx(i,j,k)
    enddo; enddo
    do J=Jsq-1,Jeq+1; do I=Isq-1,Ieq+1
      CS%sh_xy(I,J,k) = sh_xy(I,J,k) - CS%sh_xy(I,J,k)
      CS%vort_xy(I,J,k) = vort_xy(I,J,k) - CS%vort_xy(I,J,k)
    enddo; enddo
  enddo
  call cpu_clock_end(CS%id_clock_filter)

  if (.not. G%symmetric) &
    call do_group_pass(CS%pass_xy, G%Domain, &
      clock=CS%id_clock_mpi)

end subroutine filter_velocity_gradients

!> Filtering of the stress tensor Txx, Tyy, Txy.
!! The algorithm: marching halo with non-blocking grouped MPI
!! exchanges. The input arrays (Txx, Tyy, Txy) must have halo 1
!! with zero B.C. applied. The output have the same halo and B.C.
subroutine filter_stress(G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  integer :: Txx_halo, Tyy_halo, Txy_halo ! currently available halo for stress components
  integer :: Txx_iter, Tyy_iter, Txy_iter ! remaining number of iterations
  integer :: niter                        ! required number of iterations

  niter = CS%Stress_iter

  if (niter == 0) return

  Txx_halo = 1; Tyy_halo = 1; Txy_halo = 1; ! these are required halo for Txx, Tyy, Txy
  Txx_iter = niter; Tyy_iter = niter; Txy_iter = niter;

  do while &
      (Txx_iter >  0 .or. Txy_iter >  0 .or. & ! filter iterations remain to be done
       Txx_halo < 1 .or. Txy_halo < 1)         ! there is no halo for Txx or Txy

    ! ---------- filtering Txy -----------
    if (Txy_halo < 1) then
      call complete_group_pass(CS%pass_Tq, G%Domain, clock=CS%id_clock_mpi)
      Txy_halo = CS%Stress_halo
    endif

    call filter_hq(G, GV, CS, Txy_halo, Txy_iter, q=CS%Txy)

    if (Txy_halo < 1) &
       call start_group_pass(CS%pass_Tq, G%Domain, clock=CS%id_clock_mpi)

    ! ------- filtering Txx, Tyy ---------
    if (Txx_halo < 1) then
      call complete_group_pass(CS%pass_Th, G%Domain, clock=CS%id_clock_mpi)
      Txx_halo = CS%Stress_halo; Tyy_halo = CS%Stress_halo
    endif

    call filter_hq(G, GV, CS, Txx_halo, Txx_iter, h=CS%Txx)
    call filter_hq(G, GV, CS, Tyy_halo, Tyy_iter, h=CS%Tyy)

    if (Txx_halo < 1) &
      call start_group_pass(CS%pass_Th, G%Domain, clock=CS%id_clock_mpi)

  enddo

end subroutine filter_stress

!> Wrapper for filter_3D function. The border indices for q and h
!! arrays are substituted.
subroutine filter_hq(G, GV, CS, current_halo, remaining_iterations, q, h)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS      !< ZB2020 control structure.
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), optional,   &
           intent(inout) :: h !< Input/output array in h points [arbitrary]
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)), optional, &
           intent(inout) :: q !< Input/output array in q points [arbitrary]
  integer, intent(inout) :: current_halo         !< Currently available halo points
  integer, intent(inout) :: remaining_iterations !< The number of iterations to perform

  logical :: direction ! The direction of the first 1D filter

  direction = (MOD(G%first_direction,2) == 0)

  call cpu_clock_begin(CS%id_clock_filter)

  if (present(h)) then
    call filter_3D(h, CS%maskw_h,                  &
              G%isd, G%ied, G%jsd, G%jed,          &
              G%isc, G%iec, G%jsc, G%jec, GV%ke,   &
              current_halo, remaining_iterations,  &
              direction)
  endif

  if (present(q)) then
    call filter_3D(q, CS%maskw_q,                  &
            G%IsdB, G%IedB, G%JsdB, G%JedB,        &
            G%IscB, G%IecB, G%JscB, G%JecB, GV%ke, &
            current_halo, remaining_iterations,    &
            direction)
  endif

  call cpu_clock_end(CS%id_clock_filter)
end subroutine filter_hq

!> Spatial lateral filter applied to 3D array. The lateral filter is given
!! by the convolutional kernel:
!!     [1 2 1]
!! C = |2 4 2| * 1/16
!!     [1 2 1]
!! The fast algorithm decomposes the 2D filter into two 1D filters as follows:
!!     [1]
!! C = |2| * [1 2 1] * 1/16
!!     [1]
!! The input array must have zero B.C. applied. B.C. is applied for output array.
!! Note that maskw contains both land mask and 1/16 factor.
!! Filter implements marching halo. The available halo is specified and as many
!! filter iterations as possible and as needed are performed.
subroutine filter_3D(x, maskw, isd, ied, jsd, jed, is, ie, js, je, nz, &
                     current_halo, remaining_iterations,               &
                     direction)
  integer, intent(in) :: isd !< Indices of array size
  integer, intent(in) :: ied !< Indices of array size
  integer, intent(in) :: jsd !< Indices of array size
  integer, intent(in) :: jed !< Indices of array size
  integer, intent(in) :: is  !< Indices of owned points
  integer, intent(in) :: ie  !< Indices of owned points
  integer, intent(in) :: js  !< Indices of owned points
  integer, intent(in) :: je  !< Indices of owned points
  integer, intent(in) :: nz  !< Vertical array size
  real, dimension(isd:ied,jsd:jed,nz), &
           intent(inout) :: x !< Input/output array [arbitrary]
  real, dimension(isd:ied,jsd:jed), &
           intent(in) :: maskw !< Mask array of land points divided by 16 [nondim]
  integer, intent(inout) :: current_halo         !< Currently available halo points
  integer, intent(inout) :: remaining_iterations !< The number of iterations to perform
  logical, intent(in)    :: direction            !< The direction of the first 1D filter

  real, parameter :: weight = 2. ! Filter weight [nondim]
  integer :: i, j, k, iter, niter, halo

  real :: tmp(isd:ied, jsd:jed) ! Array with temporary results [arbitrary]

  ! Do as many iterations as needed and possible
  niter = min(current_halo, remaining_iterations)
  if (niter == 0) return ! nothing to do

  ! Update remaining iterations
  remaining_iterations = remaining_iterations - niter
  ! Update halo information
  current_halo = current_halo - niter

  do k=1,Nz
    halo = niter-1 + &
      current_halo ! Save as many halo points as possible
    do iter=1,niter

      if (direction) then
        do j = js-halo, je+halo; do i = is-halo-1, ie+halo+1
          tmp(i,j) = weight * x(i,j,k) + (x(i,j-1,k) + x(i,j+1,k))
        enddo; enddo

        do j = js-halo, je+halo; do i = is-halo, ie+halo;
          x(i,j,k) = (weight * tmp(i,j) + (tmp(i-1,j) + tmp(i+1,j))) * maskw(i,j)
        enddo; enddo
      else
        do j = js-halo-1, je+halo+1; do i = is-halo, ie+halo
          tmp(i,j) = weight * x(i,j,k) + (x(i-1,j,k) + x(i+1,j,k))
        enddo; enddo

        do j = js-halo, je+halo; do i = is-halo, ie+halo;
          x(i,j,k) = (weight * tmp(i,j) + (tmp(i,j-1) + tmp(i,j+1))) * maskw(i,j)
        enddo; enddo
      endif

      halo = halo - 1
    enddo
  enddo

end subroutine filter_3D

!> Computes the 3D energy source term for the ZB2020 scheme
!! similarly to MOM_diagnostics.F90, specifically 1125 line.
subroutine compute_energy_source(u, v, h, fx, fy, G, GV, CS, KE_term, global_integral)
  type(ocean_grid_type),         intent(in)  :: G    !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)  :: GV   !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(in)  :: CS   !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in)    :: u  !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in)    :: v  !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                                 intent(in) :: h     !< Layer thicknesses [H ~> m or kg m-2].

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in) :: fx    !< Zonal acceleration due to convergence of
                                                     !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in) :: fy    !< Meridional acceleration due to convergence
                                                     !! of along-coordinate stress tensor [L T-2 ~> m s-2]

  real, intent(out) :: KE_term(SZI_(G),SZJ_(G),SZK_(GV)) !< A term in the kinetic energy budget
                                                         ! [H L2 T-3 ~> m3 s-3 or W m-2]
  
  real, optional, intent(out) :: global_integral         !< Global integral of the energy effect of ZB2020
                                                         ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]

  real :: KE_u(SZIB_(G),SZJ_(G))            ! The area integral of a KE term in a layer at u-points
                                            ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]
  real :: KE_v(SZI_(G),SZJB_(G))            ! The area integral of a KE term in a layer at v-points
                                            ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]

  real :: tmp(SZI_(G),SZJ_(G),SZK_(GV))     ! temporary array for integration

  real :: uh                                ! Transport through zonal faces = u*h*dy,
                                            ! [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: vh                                ! Transport through meridional faces = v*h*dx,
                                            ! [H L2 T-1 ~> m3 s-1 or kg s-1].

  type(group_pass_type) :: pass_KE_uv       ! A handle used for group halo passes

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k

  call cpu_clock_begin(CS%id_clock_source)
  call create_group_pass(pass_KE_uv, KE_u, KE_v, G%Domain, To_North+To_East)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  KE_term(:,:,:) = 0.
  tmp(:,:,:) = 0.
  ! Calculate the KE source from Zanna-Bolton2020 [H L2 T-3 ~> m3 s-3].
  do k=1,nz
    KE_u(:,:) = 0.
    KE_v(:,:) = 0.
    do j=js,je ; do I=Isq,Ieq
      uh = u(I,j,k) * 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i+1,j)*h(i+1,j,k)) * &
        G%dyCu(I,j)
      KE_u(I,j) = uh * G%dxCu(I,j) * fx(I,j,k)
    enddo ; enddo
    do J=Jsq,Jeq ; do i=is,ie
      vh = v(i,J,k) * 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i,j+1)*h(i,j+1,k)) * &
        G%dxCv(i,J)
      KE_v(i,J) = vh * G%dyCv(i,J) * fy(i,J,k)
    enddo ; enddo
    call do_group_pass(pass_KE_uv, G%domain)
    do j=js,je ; do i=is,ie
      KE_term(i,j,k) = 0.5 * G%IareaT(i,j) &
          * (KE_u(I,j) + KE_u(I-1,j) + KE_v(i,J) + KE_v(i,J-1))
      
      if (present(global_integral)) &
        tmp(i,j,k) = KE_term(i,j,k) * G%areaT(i,j) * G%mask2dT(i,j)
    enddo ; enddo
  enddo
  
  if (present(global_integral)) &
    global_integral = reproducing_sum(tmp)

  call cpu_clock_end(CS%id_clock_source)

end subroutine compute_energy_source

end module MOM_Zanna_Bolton