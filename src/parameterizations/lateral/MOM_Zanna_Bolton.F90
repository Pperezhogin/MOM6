! > Calculates Zanna and Bolton 2020 parameterization
module MOM_Zanna_Bolton

use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type, &
                              start_group_pass, complete_group_pass
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_error_handler, only : MOM_error, WARNING, FATAL
use MOM_cpu_clock,     only : cpu_clock_id, cpu_clock_begin, cpu_clock_end
use MOM_cpu_clock,     only : CLOCK_MODULE, CLOCK_ROUTINE

implicit none ; private

#include <MOM_memory.h>

public Zanna_Bolton_2020, ZB_2020_init, ZB_2020_end, ZB_copy_gradient_and_thickness

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

  real, dimension(:,:,:), allocatable :: &
          sh_xx,   & !< Horizontal tension (du/dx - dv/dy) in h (CENTER)
                     !! points including metric terms [T-1 ~> s-1]
          sh_xy,   & !< Horizontal shearing strain (du/dy + dv/dx) in q (CORNER)
                     !! points including metric terms [T-1 ~> s-1]
          vort_xy, & !< Vertical vorticity (dv/dx - du/dy) in q (CORNER)
                     !! points including metric terms [T-1 ~> s-1]
          hq         !< Thickness in CORNER points [H ~> m or kg m-2]
  
  real, dimension(:,:,:), allocatable :: &
          Txx,     & !< Subgrid stress xx component in h [H L2 T-2 ~> m3 s-2 or kg s-2]
          Tyy,     & !< Subgrid stress yy component in h [H L2 T-2 ~> m3 s-2 or kg s-2]
          Txy        !< Subgrid stress xy component in q [H L2 T-2 ~> m3 s-2 or kg s-2]

  real, dimension(:,:), allocatable :: &
          kappa_h, & !< Scaling coefficient in h points [L2 ~> m2]
          kappa_q    !< Scaling coefficient in q points [L2 ~> m2]

  real, allocatable ::    &
        ICoriolis_h(:,:), &  !< Inverse Coriolis parameter at h points [T ~> s]
        c_diss(:,:,:)        !< Attenuation parameter at h points 
                             !! (Klower 2018, Juricke2019,2020) [nondim]

  real, dimension(:,:), allocatable ::    &
        maskw_h,  & !< Mask of land point at h points multiplied by filter weight
        maskw_q     !< Same mask but for q points

  type(diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_ZB2020u = -1, id_ZB2020v = -1, id_KE_ZB2020 = -1
  integer :: id_Txx = -1
  integer :: id_Tyy = -1
  integer :: id_Txy = -1
  integer :: id_cdiss = -1
  ! Debug fields
  integer :: id_sh_xx = -1, id_sh_xy = -1, id_vort_xy = -1
  integer :: id_hq = -1
  integer :: id_h = -1, id_u = -1, id_v = -1
  !>@}

  !>@{ CPU time clock IDs
  integer :: id_clock_module
  integer :: id_clock_copy
  integer :: id_clock_cdiss
  integer :: id_clock_stress
  integer :: id_clock_divergence
  integer :: id_clock_mpi
  integer :: id_clock_filter
  integer :: id_clock_upd
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

!> Read parameters and register output fields
!! used in Zanna_Bolton_2020().
subroutine ZB_2020_init(Time, G, GV, US, param_file, diag, CS, use_ZB2020)
  type(time_type),         intent(in)    :: Time       !< The current model time.
  type(ocean_grid_type),   intent(in)    :: G          !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV         !< The ocean's vertical grid structure
  type(unit_scale_type),   intent(in)    :: US         !< A dimensional unit scaling type
  type(param_file_type),   intent(in)    :: param_file !< Parameter file parser structure.
  type(diag_ctrl), target, intent(inout) :: diag       !< Diagnostics structure.
  type(ZB2020_CS),         intent(inout) :: CS         !< ZB2020 control structure.
  logical,                 intent(out)   :: use_ZB2020 !< If true, turns on ZB scheme.

  real :: subroundoff_Cor     !> A negligible parameter which avoids division by zero 
                              !! but small compared to Coriolis parameter [T-1 ~> s-1]

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

  call get_param(param_file, mdl, "ZB_SCALING", CS%amplitude, &
                 "The nondimensional scaling factor in ZB model, " //&
                 "typically 0.1 - 10.", units="nondim", default=0.3)

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
                 "Subgrid stress is multiplied by 1/(1+(shear/(f*R_diss))) " //&
                 "R_diss=-1: attenuation is not used; typical value R_diss=1.0", &
                 units="nondim", default=-1.)

  call get_param(param_file, mdl, "ZB_KLOWER_SHEAR", CS%Klower_shear, &
                 "Type of expression for shear in Klower formula: " //&
                 "0: sqrt(sh_xx**2 + sh_xy**2) " //&
                 "1: sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)", &
                 default=1)

  call get_param(param_file, mdl, "ZB_MARCHING_HALO", CS%Marching_halo, &
                 "The number of filter iterations per single MPI " //&
                 " exchange", default=4)

  allocate(CS%sh_xx(SZI_(G),SZJ_(G),SZK_(GV))); CS%sh_xx(:,:,:) = 0.
  allocate(CS%sh_xy(SZIB_(G),SZJB_(G),SZK_(GV))); CS%sh_xy(:,:,:) = 0.
  allocate(CS%vort_xy(SZIB_(G),SZJB_(G),SZK_(GV))); CS%vort_xy(:,:,:) = 0.
  allocate(CS%hq(SZIB_(G),SZJB_(G),SZK_(GV))); CS%hq(:,:,:) = 0.

  allocate(CS%Txx(SZI_(G),SZJ_(G),SZK_(GV))); CS%Txx(:,:,:) = 0.
  allocate(CS%Tyy(SZI_(G),SZJ_(G),SZK_(GV))); CS%Tyy(:,:,:) = 0.
  allocate(CS%Txy(SZIB_(G),SZJB_(G),SZK_(GV))); CS%Txy(:,:,:) = 0.
  allocate(CS%kappa_h(SZI_(G),SZJ_(G))); CS%kappa_h(:,:) = 0.
  allocate(CS%kappa_q(SZIB_(G),SZJB_(G))); CS%kappa_q(:,:) = 0.

  ! The scaling coefficient is multiplied to mask
  ! to provide boundary conditions
  do j=js-1,je+1 ; do i=is-1,ie+1
    CS%kappa_h(i,j) = - CS%amplitude * G%areaT(i,j) * G%mask2dT(i,j)
  enddo; enddo

  do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
    CS%kappa_q(I,J) = - CS%amplitude * G%areaBu(I,J) * G%mask2dBu(I,J)
  enddo; enddo

  if (CS%Klower_R_diss > 0) then
    allocate(CS%ICoriolis_h(SZI_(G),SZJ_(G))); CS%ICoriolis_h(:,:) = 0.
    allocate(CS%c_diss(SZI_(G),SZJ_(G),SZK_(GV))); CS%c_diss(:,:,:) = 0.

    subroundoff_Cor = 1e-30 * US%T_to_s
    do j=js-1,je+1 ; do i=is-1,ie+1
      CS%ICoriolis_h(i,j) = 1. / ((abs(0.25 * ((G%CoriolisBu(I,J) + G%CoriolisBu(I-1,J-1)) &
                          + (G%CoriolisBu(I-1,J) + G%CoriolisBu(I,J-1)))) + subroundoff_Cor) &
                          * CS%Klower_R_diss)
    enddo; enddo
  endif

  if (CS%Stress_iter > 0 .or. CS%HPF_iter > 0) then
    ! 0.0625 is the filter weight
    allocate(CS%maskw_h(SZI_(G),SZJ_(G))); CS%maskw_h(:,:) = G%mask2dT(:,:) * 0.0625
    allocate(CS%maskw_q(SZIB_(G),SZJB_(G))); CS%maskw_q(:,:) = G%mask2dBu(:,:) * 0.0625
  endif

  if (CS%Stress_iter > 0) then
    ! reduce size of halo exchange accrodingly to
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
    ! inputs of function compute_stress
    CS%HPF_halo = max(min(CS%Marching_halo, CS%HPF_iter, &
                          G%Domain%nihalo, G%Domain%njhalo), 2)

    call create_group_pass(CS%pass_xx, CS%sh_xx, G%Domain, halo=CS%HPF_halo)
    call create_group_pass(CS%pass_xy, CS%sh_xy, G%Domain, halo=CS%HPF_halo, &
      position=CORNER)
    call create_group_pass(CS%pass_xy, CS%vort_xy, G%Domain, halo=CS%HPF_halo, &
      position=CORNER)
  endif

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
      'Diagonal term (Txx) in the ZB stress tensor', 'm2s-2', conversion=US%L_T_to_m_s**2)

  CS%id_Tyy = register_diag_field('ocean_model', 'Tyy', diag%axesTL, Time, &
      'Diagonal term (Tyy) in the ZB stress tensor', 'm2s-2', conversion=US%L_T_to_m_s**2)

  CS%id_Txy = register_diag_field('ocean_model', 'Txy', diag%axesBL, Time, &
      'Off-diagonal term (Txy) in the ZB stress tensor', 'm2s-2', conversion=US%L_T_to_m_s**2)

  if (CS%Klower_R_diss > 0) then
    CS%id_cdiss = register_diag_field('ocean_model', 'c_diss', diag%axesTL, Time, &
        'Klower (2018) attenuation coefficient', '1', conversion=1.)
  endif

  ! These fields are used for debug only
  CS%id_sh_xx = register_diag_field('ocean_model', 'sh_xx', diag%axesTL, Time, &
      'Horizontal tension (du/dx - dv/dy) in h (CENTER) points including metric terms', &
      's-1', conversion=US%s_to_T)
  CS%id_sh_xy = register_diag_field('ocean_model', 'sh_xy', diag%axesBL, Time, &
      'Horizontal shearing strain (du/dy + dv/dx) in q (CORNER) points including metric terms', &
      's-1', conversion=US%s_to_T)
  CS%id_vort_xy = register_diag_field('ocean_model', 'vort_xy', diag%axesBL, Time, &
      'Vertical vorticity (dv/dx - du/dy) in q (CORNER) points including metric terms', &
      's-1', conversion=US%s_to_T)
  CS%id_hq = register_diag_field('ocean_model', 'hq', diag%axesBL, Time, &
      'Thickness in CORNER points', 'm', conversion=GV%H_to_m)
  CS%id_h = register_diag_field('ocean_model', 'h_ZB', diag%axesTL, Time, &
      'Thickness in ZB module', 'm', conversion=GV%H_to_m)
  CS%id_u = register_diag_field('ocean_model', 'u_ZB', diag%axesCuL, Time, &
      'Zonal velocity in ZB module', 'ms-1', conversion=US%L_T_to_m_s)
  CS%id_v = register_diag_field('ocean_model', 'v_ZB', diag%axesCvL, Time, &
      'Meridional velocity in ZB module', 'ms-1', conversion=US%L_T_to_m_s)

  ! Clock IDs
  ! Only module is measured with syncronization. While smaller 
  ! parts are measured without -- because these are nested clocks.
  CS%id_clock_module = cpu_clock_id('(Ocean Zanna-Bolton-2020)', grain=CLOCK_MODULE, sync=.true.)
  CS%id_clock_copy = cpu_clock_id('(ZB2020 copy fields)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_cdiss = cpu_clock_id('(ZB2020 compute c_diss)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_stress = cpu_clock_id('(ZB2020 compute stress)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_divergence = cpu_clock_id('(ZB2020 compute divergence)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_mpi = cpu_clock_id('(ZB2020 filter MPI exchanges)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_filter = cpu_clock_id('(ZB2020 filter no MPI)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_upd = cpu_clock_id('(ZB2020 update diffu, diffv)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_post = cpu_clock_id('(ZB2020 post data)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_source = cpu_clock_id('(ZB2020 compute energy source)', grain=CLOCK_ROUTINE, sync=.false.)

end subroutine ZB_2020_init

!> Deallocated any variables allocated in ZB_2020_init
subroutine ZB_2020_end(CS)
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

  if (CS%Stress_iter > 0 .or. CS%HPF_iter) then
    deallocate(CS%maskw_h)
    deallocate(CS%maskw_q)
  endif

end subroutine ZB_2020_end

!> Save precomputed velocity gradients and thicknesses
!! from the horizontal eddy viscosity module
!! We save sh_xx with halo 2 (is-2:ie+2,js-2:je+2)
!! sh_xy and vort_xy with halo 1 (Isq-1:Ieq+1,Jsq-1:Jeq+1)
!! These are maximum halo sizes possible with halo 2 
!! used for velocities u, v in RK2 stepping routine
subroutine ZB_copy_gradient_and_thickness(sh_xx, sh_xy, vort_xy, hq, &
                                       G, GV, CS, k)
  type(ocean_grid_type),         intent(in)    :: G      !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)    :: GV     !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(inout) :: CS     !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(in) :: sh_xy,   &  !< horizontal shearing strain (du/dy + dv/dx) 
                              !! including metric terms [T-1 ~> s-1]
                  vort_xy, &  !< Vertical vorticity (dv/dx - du/dy) 
                              !! including metric terms [T-1 ~> s-1]
                  hq          !< harmonic mean of the harmonic means 
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

end subroutine ZB_copy_gradient_and_thickness

!> Baroclinic Zanna-Bolton-2020 parameterization, see
!! eq. 6 in https://laurezanna.github.io/files/Zanna-Bolton-2020.pdf
subroutine Zanna_Bolton_2020(u, v, h, diffu, diffv, G, GV, CS, &
                             dx2h, dy2h, dx2q, dy2q)
  type(ocean_grid_type),         intent(in)    :: G      !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)    :: GV     !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(inout) :: CS     !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in)    :: u    !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in)    :: v    !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                                 intent(in) :: h       !< Layer thicknesses [H ~> m or kg m-2].

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(inout) :: diffu   !< Zonal acceleration due to convergence of
                                                          !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(inout) :: diffv   !< Meridional acceleration due to convergence
                                                          !! of along-coordinate stress tensor [L T-2 ~> m s-2]

  real, dimension(SZI_(G),SZJ_(G)),           & 
                                 intent(in) :: dx2h, & !< dx^2 at h points [L2 ~> m2]
                                               dy2h    !< dy^2 at h points [L2 ~> m2]

  real, dimension(SZIB_(G),SZJB_(G)),           & 
                                 intent(in) :: dx2q, & !< dx^2 at q points [L2 ~> m2]
                                               dy2q    !< dy^2 at q points [L2 ~> m2]

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)) :: &
    ZB2020u           !< Zonal acceleration due to convergence of
                      !! along-coordinate stress tensor for ZB model
                      !! [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)) :: &
    ZB2020v           !< Meridional acceleration due to convergence
                      !! of along-coordinate stress tensor for ZB model
                      !! [L T-2 ~> m s-2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  call cpu_clock_begin(CS%id_clock_module)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  if (CS%Klower_R_diss > 0.) then
    call compute_c_diss(CS%sh_xx, CS%sh_xy, CS%vort_xy, &
                        CS%c_diss, G, GV, CS)
  endif

  call filter_velocity_gradients(G, GV, CS)

  call compute_stress(CS%sh_xx, CS%sh_xy, CS%vort_xy,   & 
                      CS%Txx, CS%Tyy, CS%Txy,           &
                      G, GV, CS)

  call filter_stress(G, GV, CS)

  call compute_stress_divergence(CS%Txx, CS%Tyy, CS%Txy, h, &
                                 ZB2020u, ZB2020v, G, GV, CS, &
                                 dx2h, dy2h, dx2q, dy2q)

  call cpu_clock_begin(CS%id_clock_upd)
  do k=1,nz ; do j=js,je ; do I=Isq,Ieq
    diffu(I,j,k) = diffu(I,j,k) + ZB2020u(I,j,k)
  enddo ; enddo ; enddo

  do k=1,nz ; do J=Jsq,Jeq ; do i=is,ie
    diffv(i,J,k) = diffv(i,J,k) + ZB2020v(i,J,k)
  enddo ; enddo ; enddo
  call cpu_clock_end(CS%id_clock_upd)

  call cpu_clock_begin(CS%id_clock_post)
  ! Acceleration
  if (CS%id_ZB2020u>0)   call post_data(CS%id_ZB2020u, ZB2020u, CS%diag)
  if (CS%id_ZB2020v>0)   call post_data(CS%id_ZB2020v, ZB2020v, CS%diag)

  ! Stress tensor
  if (CS%id_Txx>0)     call post_data(CS%id_Txx, CS%Txx, CS%diag)
  if (CS%id_Tyy>0)     call post_data(CS%id_Tyy, CS%Tyy, CS%diag)
  if (CS%id_Txy>0)     call post_data(CS%id_Txy, CS%Txy, CS%diag)

  ! Klower attenuation
  if (CS%id_cdiss>0)     call post_data(CS%id_cdiss, CS%c_diss, CS%diag)

  ! Debug fields
  if (CS%id_sh_xx>0)   call post_data(CS%id_sh_xx, CS%sh_xx, CS%diag)
  if (CS%id_sh_xy>0)   call post_data(CS%id_sh_xy, CS%sh_xy, CS%diag)
  if (CS%id_vort_xy>0) call post_data(CS%id_vort_xy, CS%vort_xy, CS%diag)
  if (CS%id_hq>0)      call post_data(CS%id_hq, CS%hq, CS%diag)
  if (CS%id_h>0)       call post_data(CS%id_h, h, CS%diag)
  if (CS%id_u>0)       call post_data(CS%id_u, u, CS%diag)
  if (CS%id_v>0)       call post_data(CS%id_v, v, CS%diag)
  call cpu_clock_end(CS%id_clock_post)

  call compute_energy_source(u, v, h, ZB2020u, ZB2020v, G, GV, CS)

  call cpu_clock_end(CS%id_clock_module)

end subroutine Zanna_Bolton_2020

!> Compute the attenuation parameter similarly
!! in h points as in Klower2018, Juricke2019,2020:
!! c_diss = 1/(1+(shear/(abs(f)*R_diss)))
!! where shear = sqrt(sh_xx**2 + sh_xy**2)
!! or shear = sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)
!! The arrays sh_xx, sh_xy, vort_xy should have halo 1, i.e.
!! (is-1:ie+1,js-1:je+1) and (Isq-1:Ieq+1,Jsq-1:Jeq+1)
!! B.C. is required for sh_xy and vort_xy (for interpolation)
!! B.C. for sh_xx is not needed
subroutine compute_c_diss(sh_xx, sh_xy, vort_xy, c_diss, G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS   !< ZB2020 control structure.

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),   &
        intent(in) :: sh_xx     !< Horizontal tension (du/dx - dv/dy) in h (CENTER)
                                !! points including metric terms [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)), &
        intent(in) :: sh_xy, &  !< Horizontal shearing strain (du/dy + dv/dx) in q (CORNER)
                                !! points including metric terms [T-1 ~> s-1]
                      vort_xy   !< Vertical vorticity (dv/dx - du/dy) in q (CORNER)
                                !! points including metric terms [T-1 ~> s-1]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),   &
        intent(inout) :: c_diss !< Attenuation parameter in h points
                                !! (Klower 2018, Juricke2019,2020) [nondim]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  real :: shear ! Shear in Klower2018 formula at h points [L-1 ~> s-1]

  call cpu_clock_begin(CS%id_clock_cdiss)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  do k=1,nz
    do j=js-1,je+1 ; do i=is-1,ie+1
      ! sqrt(sh_xx**2 + sh_xy**2)
      if (CS%Klower_shear == 0) then
        shear = sqrt(sh_xx(i,j,k)**2 + 0.25 * (          &
          (sh_xy(I-1,J-1,k)**2 + sh_xy(I,J  ,k)**2)      &
        + (sh_xy(I-1,J  ,k)**2 + sh_xy(I,J-1,k)**2)      &
        ))
      ! sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)
      elseif (CS%Klower_shear == 1) then
        shear = sqrt(sh_xx(i,j,k)**2 + 0.25 * (          &
          ((sh_xy(I-1,J-1,k)**2+vort_xy(I-1,J-1,k)**2)   &
         + (sh_xy(I,J,k)**2+vort_xy(I,J,k)**2))          &
        + ((sh_xy(I-1,J,k)**2+vort_xy(I-1,J,k)**2)       &
         + (sh_xy(I,J-1,k)**2+vort_xy(I,J-1,k)**2))      &
        ))
      endif
      
      c_diss(i,j,k) = 1. / (1. + shear * CS%ICoriolis_h(i,j))

    enddo; enddo

  enddo ! end of k loop
  
  call cpu_clock_end(CS%id_clock_cdiss)

end subroutine compute_c_diss

!> Compute stress tensor T
!! (Txx, Txy;
!!  Txy, Tyy)
!! Which consists of the deviatoric and trace components, respectively:
!! T =   (-vort_xy * sh_xy, vort_xy * sh_xx;
!!         vort_xy * sh_xx,  vort_xy * sh_xy) +
!! 1/2 * (vort_xy^2 + sh_xy^2 + sh_xx^2, 0;
!!        0, vort_xy^2 + sh_xy^2 + sh_xx^2)
!! Trace is multiplied by kappa=-ampiltude * grid_cell_area:
!! T -> T * kappa
!! Update of the governing equations:
!! (du/dt, dv/dt) = div(T)
!! The input arrays vort_xy and sh_xy should have zero boundary
!! condtions. 
!! B.C. for sh_xx is not required
!! Arrays vort_xy and sh_xy should have 1 halo,
!! i.e. (Isq-1:Ieq+1, Jsq-1:Jeq+1).
!! Arrays sh_xx also should have 1 halo,
!! i.e. (is-1:ie+1, js-1:je+1)
!! If conditions are satisfied, the output arrays Txx, Tyy
!! will have halo 1 (is-1:is+1, js-1:js+1) and Txy
!! will have halo 0 (Isq:Ieq, Jsq:Jeq), while
!! Txy will have applied B.C. 
!! B.C. for Txx, Tyy is not applied.
!! B.C. for Txx, Tyy is not needed for ZB2020,
!! but may be needed for filtering.
subroutine compute_stress(sh_xx, sh_xy, vort_xy, Txx, Tyy, Txy, G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS   !< ZB2020 control structure.

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),   &
        intent(in) :: sh_xx     !< Horizontal tension (du/dx - dv/dy) in h (CENTER)
                                !! points including metric terms [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)), &
        intent(in) :: sh_xy, &  !< Horizontal shearing strain (du/dy + dv/dx) in q (CORNER)
                                !! points including metric terms [T-1 ~> s-1]
                      vort_xy   !< Vertical vorticity (dv/dx - du/dy) in q (CORNER)
                                !! points including metric terms [T-1 ~> s-1]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),     &
        intent(inout) :: Txx,     & !< Subgrid stress xx component in h [L2 T-2 ~> m2 s-2]
                         Tyy        !< Subgrid stress yy component in h [L2 T-2 ~> m2 s-2]
  
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)),   &
        intent(inout) :: Txy        !< Subgrid stress xy component in q [L2 T-2 ~> m2 s-2]

  real :: &
    vort_xy_h, &  ! Vorticity interpolated to the h point [T-1 ~> s-1]
    sh_xy_h       ! Shearing strain interpolated to the h point [T-1 ~> s-1]

  real :: &
    sh_xx_q       ! Horizontal tension interpolated to the q point [T-1 ~> s-1]

  real :: sum_sq       ! 1/2*(vort_xy^2 + sh_xy^2 + sh_xx^2) [T-2 ~> s-2]
  real :: vort_sh      ! vort_xy*sh_xy [T-2 ~> s-2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  call cpu_clock_begin(CS%id_clock_stress)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  sum_sq = 0.
  vort_sh = 0.

  do k=1,nz

    ! compute Txx, Tyy tensor
    do j=js-1,je+1 ; do i=is-1,ie+1
      ! It is assumed that B.C. is applied to sh_xy and vort_xy
      sh_xy_h = 0.25 * ( (sh_xy(I-1,J-1,k) + sh_xy(I,J,k)) &
                       + (sh_xy(I-1,J,k) + sh_xy(I,J-1,k)) )

      vort_xy_h = 0.25 * ( (vort_xy(I-1,J-1,k) + vort_xy(I,J,k)) &
                         + (vort_xy(I-1,J,k) + vort_xy(I,J-1,k)) ) 

      if (CS%ZB_type .NE. 1) then
        sum_sq = 0.5 *                    &
          (vort_xy_h * vort_xy_h          &
           + sh_xy_h * sh_xy_h            &
           + sh_xx(i,j,k) * sh_xx(i,j,k)  &
            )
      endif

      if (CS%ZB_type .NE. 2) then
        if (CS%ZB_cons == 0) then
          vort_sh = vort_xy_h * sh_xy_h
        else if (CS%ZB_cons == 1) then
          ! It is assumed that B.C. is applied to sh_xy and vort_xy
          vort_sh = 0.25 * (                                                  &
                (G%areaBu(I-1,J-1) * vort_xy(I-1,J-1,k) * sh_xy(I-1,J-1,k)  + &
                 G%areaBu(I  ,J  ) * vort_xy(I  ,J  ,k) * sh_xy(I  ,J  ,k)) + &
                (G%areaBu(I-1,J  ) * vort_xy(I-1,J  ,k) * sh_xy(I-1,J  ,k)  + &
                 G%areaBu(I  ,J-1) * vort_xy(I  ,J-1,k) * sh_xy(I  ,J-1,k))   &
                ) * G%IareaT(i,j)
        endif
      endif
      
      Txx(i,j,k) = CS%kappa_h(i,j) * (- vort_sh + sum_sq)
      Tyy(i,j,k) = CS%kappa_h(i,j) * (+ vort_sh + sum_sq)

    enddo ; enddo

    ! Here we assume that Txy is initialized to zero
    if (CS%ZB_type .NE. 2) then
      do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
        sh_xx_q = 0.25 * ( (sh_xx(i+1,j+1,k) + sh_xx(i,j,k)) &
                         + (sh_xx(i+1,j,k) + sh_xx(i,j+1,k)))
        ! We assume that vort_xy has zero B.C.. So,
        ! no additional masking is required
        Txy(I,J,k) = CS%kappa_q(I,J) * (vort_xy(I,J,k) * sh_xx_q)

      enddo ; enddo
    endif

  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_stress)

end subroutine compute_stress

!> Compute the divergence of subgrid stress
!! weghted with thickness, i.e.
!! (fx,fy) = 1/h Div(h * [Txx, Txy; Txy, Tyy])
!! Optionally, before the divergence, we attenuate the stress 
!! according to the Klower formula
!! Txx, Tyy, c_diss should have halo 1 (is-1:is+1, js-1:js+1)
!! Txy should be in q points with halo 0 (Isq:Ieq, Jsq:Jeq)
!! B.C. for Txy is required, while for 
!! Txx, Tyy, c_diss is not required
subroutine compute_stress_divergence(Txx, Tyy, Txy, h, fx, fy, G, GV, CS, &
                                     dx2h, dy2h, dx2q, dy2q)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS   !< ZB2020 control structure.

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),     &
        intent(inout) :: Txx,     & !< Subgrid stress xx component in h;
                                    !! Input:[L2 T-2 ~> m2 s-2], Output: 
                                    !!       [H L2 T-2 ~> m3 s-2 or kg s-2]
                                    !! And same for tensors Tyy, Txy below
                         Tyy        !< Subgrid stress yy component in h [L2 T-2 ~> m2 s-2]
  
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)),   &
        intent(inout) :: Txy        !< Subgrid stress xy component in q [L2 T-2 ~> m2 s-2]
  
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
        intent(in) :: h             !< Layer thicknesses [H ~> m or kg m-2].

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
        intent(out) :: fx           !< Zonal acceleration due to convergence of
                                    !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
        intent(out) :: fy           !< Meridional acceleration due to convergence
                                    !! of along-coordinate stress tensor [L T-2 ~> m s-2]

  real, dimension(SZI_(G),SZJ_(G)),           & 
        intent(in) :: dx2h, &       !< dx^2 at h points [L2 ~> m2]
                      dy2h          !< dy^2 at h points [L2 ~> m2]

  real, dimension(SZIB_(G),SZJB_(G)),           & 
        intent(in) :: dx2q, &       !< dx^2 at q points [L2 ~> m2]
                      dy2q          !< dy^2 at q points [L2 ~> m2]

  real, dimension(SZI_(G),SZJ_(G)) :: &
        Mxx, & !< Subgrid stress xx component in h
        Myy    !< Subgrid stress yy component in h

  real, dimension(SZIB_(G),SZJB_(G)) :: &
        Mxy    !< Subgrid stress xy component in q

  real :: h_u !< Thickness interpolated to u points [H ~> m or kg m-2].
  real :: h_v !< Thickness interpolated to v points [H ~> m or kg m-2].

  real :: h_neglect    ! Thickness so small it can be lost in 
                       ! roundoff and so neglected [H ~> m or kg m-2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  call cpu_clock_begin(CS%id_clock_divergence)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  h_neglect  = GV%H_subroundoff ! Line 410 on MOM_hor_visc.F90

  do k=1,nz
    do J=js-1,Jeq ; do I=is-1,Ieq
      if (CS%Klower_R_diss > 0.) then
        Mxy(I,J) = (Txy(I,J,k) * &
                    0.25 * ((CS%c_diss(i,j  ,k) + CS%c_diss(i+1,j+1,k))   &
                          + (CS%c_diss(i,j+1,k) + CS%c_diss(i+1,j  ,k)))) &
                    * (CS%hq(I,J,k))
      else
        Mxy(I,J) = Txy(I,J,k) * (CS%hq(I,J,k))
      endif
    enddo ; enddo

    do j=js-1,je+1 ; do i=is-1,ie+1
      if (CS%Klower_R_diss > 0.) then
        Mxx(i,j) = ((Txx(i,j,k) * CS%c_diss(i,j,k)) * h(i,j,k)) * dy2h(i,j)
        Myy(i,j) = ((Tyy(i,j,k) * CS%c_diss(i,j,k)) * h(i,j,k)) * dx2h(i,j)
      else
        Mxx(i,j) = ((Txx(i,j,k)) * h(i,j,k)) * dy2h(i,j)
        Myy(i,j) = ((Tyy(i,j,k)) * h(i,j,k)) * dx2h(i,j)
      endif
    enddo ; enddo

    ! Evaluate 1/h x.Div(h S) (Line 1495 of MOM_hor_visc.F90)
    ! Minus occurs because in original file (du/dt) = - div(S),
    ! but here is the discretization of div(S)
    do j=js,je ; do I=Isq,Ieq
      h_u = 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i+1,j)*h(i+1,j,k)) + h_neglect
      fx(I,j,k) = - ((G%IdyCu(I,j)*(Mxx(i,j)                - &
                                    Mxx(i+1,j))             + &
                      G%IdxCu(I,j)*(dx2q(I,J-1)*Mxy(I,J-1)  - &
                                    dx2q(I,J)  *Mxy(I,J)))  * &
                      G%IareaCu(I,j)) / h_u
    enddo ; enddo

    ! Evaluate 1/h y.Div(h S) (Line 1517 of MOM_hor_visc.F90)
    do J=Jsq,Jeq ; do i=is,ie
      h_v = 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i,j+1)*h(i,j+1,k)) + h_neglect
      fy(i,J,k) = - ((G%IdyCv(i,J)*(dy2q(I-1,J)*Mxy(I-1,J)   - &
                                    dy2q(I,J)  *Mxy(I,J))    + & ! NOTE this plus
                      G%IdxCv(i,J)*(Myy(i,j)                 - &
                                    Myy(i,j+1)))             * &
                      G%IareaCv(i,J)) / h_v
    enddo ; enddo

  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_divergence)

end subroutine compute_stress_divergence

!> Filtering of the velocity gradients (VG)
subroutine filter_velocity_gradients(G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.

  real, dimension(SZI_(G), SZJ_(G), SZK_(GV)) :: &
        sh_xx          !< Copy of CS%sh_xx
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)) :: &
        sh_xy, vort_xy !< Copy of CS%sh_xy and CS%vort_xy

  integer :: xx_halo, xy_halo, vort_halo ! currently available halo
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

!> Filtering of the stress tensor 
!! with overlapping computations and exchanges
!! and flexible marching halo
subroutine filter_stress(G, GV, CS)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(inout) :: CS   !< ZB2020 control structure.
  
  integer :: Txx_halo, Tyy_halo, Txy_halo ! currently available halo
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

subroutine filter_hq(G, GV, CS, current_halo, remaining_iterations, q, h)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS      !< ZB2020 control structure.

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), optional,   &
           intent(inout) :: h !< Input/output array in h points [dim arbitrary]
  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)), optional, &
           intent(inout) :: q !< Input/output array in q points [dim arbitrary
  integer, intent(inout) :: current_halo, remaining_iterations

  logical :: direction

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

subroutine filter_3D(x, maskw, isd, ied, jsd, jed, is, ie, js, je, nz, &
                     current_halo, remaining_iterations,               &
                     direction)
  real, dimension(isd:ied,jsd:jed,nz), &
        intent(inout) :: x            !< Input/output array [dim arbitrary]
  real, dimension(isd:ied,jsd:jed), &
        intent(in)    :: maskw         !< Mask array of land points divded by 16 [nondim]  
  integer, intent(in) :: &
                         isd, ied,  & !< Indices of array size
                         jsd, jed,  & !< Indices of array size
                         is,  ie,   & !< Indices of owned points
                         js,  je,   &   !< Indices of owned points
                         nz
  integer, intent(inout) :: current_halo, remaining_iterations
  logical, intent(in)    :: direction
                        
  real, parameter :: two = 2.
  
  integer :: i, j, k, iter, niter, halo

  real :: tmp(isd:ied, jsd:jed)

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
          tmp(i,j) = two * x(i,j,k) + (x(i,j-1,k) + x(i,j+1,k))
        enddo; enddo
        
        do j = js-halo, je+halo; do i = is-halo, ie+halo;
          x(i,j,k) = (two * tmp(i,j) + (tmp(i-1,j) + tmp(i+1,j))) * maskw(i,j)
        enddo; enddo
      else
        do j = js-halo-1, je+halo+1; do i = is-halo, ie+halo
          tmp(i,j) = two * x(i,j,k) + (x(i-1,j,k) + x(i+1,j,k))
        enddo; enddo
        
        do j = js-halo, je+halo; do i = is-halo, ie+halo;
          x(i,j,k) = (two * tmp(i,j) + (tmp(i,j-1) + tmp(i,j+1))) * maskw(i,j)
        enddo; enddo
      endif

      halo = halo - 1
    enddo
  enddo

end subroutine filter_3D

!> Computes the 3D energy source term for the ZB2020 scheme
!! similarly to MOM_diagnostics.F90, specifically 1125 line.
subroutine compute_energy_source(u, v, h, fx, fy, G, GV, CS)
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

  real :: KE_term(SZI_(G),SZJ_(G),SZK_(GV)) ! A term in the kinetic energy budget
                                            ! [H L2 T-3 ~> m3 s-3 or W m-2]
  real :: KE_u(SZIB_(G),SZJ_(G))            ! The area integral of a KE term in a layer at u-points
                                            ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]
  real :: KE_v(SZI_(G),SZJB_(G))            ! The area integral of a KE term in a layer at v-points
                                            ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]

  !real :: tmp(SZI_(G),SZJ_(G),SZK_(GV))    ! temporary array for integration
  !real :: global_integral                  ! Global integral of the energy effect of ZB2020
                                            ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]


  real :: uh                                ! Transport through zonal faces = u*h*dy,
                                            ! [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: vh                                ! Transport through meridional faces = v*h*dx,
                                            ! [H L2 T-1 ~> m3 s-1 or kg s-1].

  type(group_pass_type) :: pass_KE_uv       ! A handle used for group halo passes

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k

  call cpu_clock_begin(CS%id_clock_source)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  if (CS%id_KE_ZB2020 > 0) then
    call create_group_pass(pass_KE_uv, KE_u, KE_v, G%Domain, To_North+To_East)

    KE_term(:,:,:) = 0.
    !tmp(:,:,:) = 0.
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
        ! copy-paste from MOM_spatial_means.F90, line 42
        !tmp(i,j,k) = KE_term(i,j,k) * G%areaT(i,j) * G%mask2dT(i,j)
      enddo ; enddo
    enddo

    !global_integral = reproducing_sum(tmp)

    call post_data(CS%id_KE_ZB2020, KE_term, CS%diag)
  endif

  call cpu_clock_end(CS%id_clock_source)

end subroutine compute_energy_source

end module MOM_Zanna_Bolton