! > Calculates Zanna and Bolton 2020 parameterization
module MOM_Zanna_Bolton

use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_error_handler, only : MOM_error, WARNING
use MOM_cpu_clock,             only : cpu_clock_id, cpu_clock_begin, cpu_clock_end
use MOM_cpu_clock,             only : CLOCK_MODULE, CLOCK_ROUTINE

implicit none ; private

#include <MOM_memory.h>

public Zanna_Bolton_2020, ZB_2020_init, ZB_2020_end, ZB_pass_gradient_and_thickness

!> Control structure for Zanna-Bolton-2020 parameterization.
type, public :: ZB2020_CS ; private
  ! Parameters
  real      :: amplitude      !< The nondimensional scaling factor in ZB model,
                              !! typically 0.1 - 10 [nondim].
  integer   :: ZB_type        !< Select how to compute the trace part of ZB model:
                              !! 0 - both deviatoric and trace components are computed
                              !! 1 - only deviatoric component is computed
                              !! 2 - only trace component is computed
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

  real :: subroundoff = 1e-30 !> A negligible parameter which avoids division by zero, but is too small to
                              !! modify physical values. [nondim]

  real, dimension(:,:,:), allocatable :: &
          sh_xx,   & !< Horizontal tension (du/dx - dv/dy) in h (CENTER)
                     !! points including metric terms [T-1 ~> s-1]
          sh_xy,   & !< Horizontal shearing strain (du/dy + dv/dx) in q (CORNER)
                     !! points including metric terms [T-1 ~> s-1]
          vort_xy, & !< Vertical vorticity (dv/dx - du/dy) in q (CORNER)
                     !! points including metric terms [T-1 ~> s-1]
          h_u,     & !< Thickness interpolated to u points [H ~> m or kg m-2]
          h_v,     & !< Thickness interpolated to v points [H ~> m or kg m-2]
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

  type(diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_ZB2020u = -1, id_ZB2020v = -1, id_KE_ZB2020 = -1
  integer :: id_Txx = -1
  integer :: id_Tyy = -1
  integer :: id_Txy = -1
  !>@}

  !>@{ CPU time clock IDs
  integer :: id_clock_module
  integer :: id_clock_pass
  integer :: id_clock_cdiss
  integer :: id_clock_stress
  integer :: id_clock_divergence
  integer :: id_clock_upd
  integer :: id_clock_post
  integer :: id_clock_source
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

  integer :: Isq, Ieq, Jsq, Jeq
  integer :: i, j

  ! This include declares and sets the variable "version".
#include "version_variable.h"
  character(len=40)  :: mdl = "MOM_Zanna_Bolton" ! This module's name.

  Isq  = G%IscB ; Ieq  = G%IecB ; Jsq  = G%JscB ; Jeq  = G%JecB

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
                 default=0)

  allocate(CS%sh_xx(SZI_(G),SZJ_(G),SZK_(GV))); CS%sh_xx(:,:,:) = 0.
  allocate(CS%sh_xy(SZIB_(G),SZJB_(G),SZK_(GV))); CS%sh_xy(:,:,:) = 0.
  allocate(CS%vort_xy(SZIB_(G),SZJB_(G),SZK_(GV))); CS%vort_xy(:,:,:) = 0.
  allocate(CS%h_u(SZIB_(G),SZJ_(G),SZK_(GV))); CS%h_u(:,:,:) = 0.
  allocate(CS%h_v(SZI_(G),SZJB_(G),SZK_(GV))); CS%h_v(:,:,:) = 0.
  allocate(CS%hq(SZIB_(G),SZJB_(G),SZK_(GV))); CS%hq(:,:,:) = 0.

  allocate(CS%Txx(SZI_(G),SZJ_(G),SZK_(GV))); CS%Txx(:,:,:) = 0.
  allocate(CS%Tyy(SZI_(G),SZJ_(G),SZK_(GV))); CS%Tyy(:,:,:) = 0.
  allocate(CS%Txy(SZIB_(G),SZJB_(G),SZK_(GV))); CS%Txy(:,:,:) = 0.
  allocate(CS%kappa_h(SZI_(G),SZJ_(G))); CS%kappa_h(:,:) = 0.
  allocate(CS%kappa_q(SZIB_(G),SZJB_(G))); CS%kappa_q(:,:) = 0.

  do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
    CS%kappa_h(i,j) = - CS%amplitude * G%areaT(i,j)
  enddo; enddo

  do J=Jsq-1,Jeq ; do I=Isq-1,Ieq
    CS%kappa_q = - CS%amplitude * G%areaBu(i,j)
  enddo; enddo

  if (CS%Klower_R_diss > 0) then
    allocate(CS%ICoriolis_h(SZI_(G),SZJ_(G))); CS%ICoriolis_h(:,:) = 0.
    allocate(CS%c_diss(SZI_(G),SZJ_(G),SZK_(GV))); CS%c_diss(:,:,:) = 0.

    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
      CS%ICoriolis_h(i,j) = 1. / ((abs(0.25 * ((G%CoriolisBu(I,J) + G%CoriolisBu(I-1,J-1)) &
                          + (G%CoriolisBu(I-1,J) + G%CoriolisBu(I,J-1)))) + CS%subroundoff) &
                          * CS%Klower_R_diss)
    enddo; enddo
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

  ! action of filter on momentum flux
  CS%id_Txx = register_diag_field('ocean_model', 'Txx', diag%axesTL, Time, &
      'Diagonal term (Txx) in the ZB stress tensor', 'm2s-2', conversion=US%L_T_to_m_s**2)

  CS%id_Tyy = register_diag_field('ocean_model', 'Tyy', diag%axesTL, Time, &
      'Diagonal term (Tyy) in the ZB stress tensor', 'm2s-2', conversion=US%L_T_to_m_s**2)

  CS%id_Txy = register_diag_field('ocean_model', 'Txy', diag%axesBL, Time, &
      'Off-diagonal term (Txy) in the ZB stress tensor', 'm2s-2', conversion=US%L_T_to_m_s**2)

  ! Clock IDs
  CS%id_clock_module = cpu_clock_id('(Ocean Zanna-Bolton-2020)', grain=CLOCK_MODULE)
  CS%id_clock_pass = cpu_clock_id('(ZB2020 pass fields)', grain=CLOCK_ROUTINE)
  CS%id_clock_cdiss = cpu_clock_id('(ZB2020 compute c_diss)', grain=CLOCK_ROUTINE)
  CS%id_clock_stress = cpu_clock_id('(ZB2020 compute stress)', grain=CLOCK_ROUTINE)
  CS%id_clock_divergence = cpu_clock_id('(ZB2020 compute divergence)', grain=CLOCK_ROUTINE)
  CS%id_clock_upd = cpu_clock_id('(ZB2020 update diffu, diffv)', grain=CLOCK_ROUTINE)
  CS%id_clock_post = cpu_clock_id('(ZB2020 post data)', grain=CLOCK_ROUTINE)
  CS%id_clock_source = cpu_clock_id('(ZB2020 compute energy source)', grain=CLOCK_ROUTINE)

end subroutine ZB_2020_init

!> Deallocated any variables allocated in ZB_2020_init
subroutine ZB_2020_end(CS)
  type(ZB2020_CS), intent(inout) :: CS  !< ZB2020 control structure.

  deallocate(CS%sh_xx)
  deallocate(CS%sh_xy)
  deallocate(CS%vort_xy)
  deallocate(CS%h_u)
  deallocate(CS%h_v)
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

end subroutine ZB_2020_end

!> Save precomputed velocity gradients and thicknesses
!! from the horizontal eddy viscosity module
subroutine ZB_pass_gradient_and_thickness(sh_xx, sh_xy, vort_xy, hq, h_u, h_v, &
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
  
  real, dimension(SZIB_(G),SZJ_(G)), &
    intent(in) :: h_u         !< Thickness interpolated to u points [H ~> m or kg m-2].
  real, dimension(SZI_(G),SZJB_(G)), &
    intent(in) :: h_v         !< Thickness interpolated to v points [H ~> m or kg m-2].

  integer, intent(in) :: k    !< The vertical index of the layer to be passed.

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  call cpu_clock_begin(CS%id_clock_pass)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  do J=js-1,Jeq ; do I=is-1,Ieq
    CS%hq(I,J,k) = hq(I,J)
  enddo; enddo

  do j=js-2,je+2 ; do I=Isq-1,Ieq+1
    CS%h_u(I,j,k) = h_u(I,j)
  enddo; enddo

  do J=Jsq-1,Jeq+1 ; do i=is-2,ie+2
    CS%h_v(i,J,k) = h_v(i,J)
  enddo; enddo

  do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
    CS%sh_xx(i,j,k) = sh_xx(i,j)
  enddo ; enddo
  
  do J=js-2,Jeq+1 ; do I=is-2,Ieq+1
    CS%sh_xy(I,J,k) = sh_xy(I,J)
  enddo; enddo

  do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
    CS%vort_xy(I,J,k) = vort_xy(I,J)
  enddo; enddo

  call cpu_clock_end(CS%id_clock_pass)

end subroutine ZB_pass_gradient_and_thickness

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

  real, dimension(SZI_(G),SZJ_(G)),           & 
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

  call compute_stress(CS%sh_xx, CS%sh_xy, CS%vort_xy,   & 
                      CS%Txx, CS%Tyy, CS%Txy,           &
                      G, GV, CS)

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
  if (CS%id_ZB2020u>0)   call post_data(CS%id_ZB2020u, ZB2020u, CS%diag)
  if (CS%id_ZB2020v>0)   call post_data(CS%id_ZB2020v, ZB2020v, CS%diag)

  if (CS%id_Txx>0)     call post_data(CS%id_Txx, CS%Txx, CS%diag)

  if (CS%id_Tyy>0)     call post_data(CS%id_Tyy, CS%Tyy, CS%diag)

  if (CS%id_Txy>0)     call post_data(CS%id_Txy, CS%Txy, CS%diag)
  call cpu_clock_end(CS%id_clock_post)

  call compute_energy_source(u, v, h, ZB2020u, ZB2020v, G, GV, CS)

  call cpu_clock_end(CS%id_clock_module)

end subroutine Zanna_Bolton_2020

!> Compute the attenuation parameter similarly
!! in h points as in Klower2018, Juricke2019,2020:
!! c_diss = 1/(1+(shear/(abs(f)*R_diss)))
!! where shear = sqrt(sh_xx**2 + sh_xy**2)
!! or shear = sqrt(sh_xx**2 + sh_xy**2 + vort_xy**2)
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

  integer :: Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  real :: shear ! Shear in Klower2018 formula at h points [L-1 ~> s-1]

  call cpu_clock_begin(CS%id_clock_cdiss)

  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB; nz = GV%ke

  do k=1,nz

    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
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

  ! Arrays defined in h (CENTER) points
  real, dimension(SZI_(G),SZJ_(G)) :: &
    vort_xy_center, &  ! Vorticity interpolated to the center [T-1 ~> s-1]
    sh_xy_center       ! Shearing strain interpolated to the center [T-1 ~> s-1]

  ! Arrays defined in q (CORNER) points
  real, dimension(SZIB_(G),SZJB_(G)) :: &
    sh_xx_corner       ! Horizontal tension interpolated to the corner [T-1 ~> s-1]

  real :: sum_sq       ! 1/2*(vort_xy^2 + sh_xy^2 + sh_xx^2) [T-2 ~> s-2]
  real :: vort_sh      ! vort_xy*sh_xy [T-2 ~> s-2]

  integer :: Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  call cpu_clock_begin(CS%id_clock_stress)

  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB; nz = GV%ke

  sum_sq = 0.
  vort_sh = 0.

  do k=1,nz

    ! interpolation
    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
      sh_xy_center(i,j) = 0.25 * ( (sh_xy(I-1,J-1,k) + sh_xy(I,J,k)) &
                                 + (sh_xy(I-1,J,k) + sh_xy(I,J-1,k)) )
    enddo ; enddo

    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
      vort_xy_center(i,j) = 0.25 * ( (vort_xy(I-1,J-1,k) + vort_xy(I,J,k)) &
                                   + (vort_xy(I-1,J,k) + vort_xy(I,J-1,k)) )
    enddo ; enddo

    do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
      sh_xx_corner(I,J) = 0.25 * ( (sh_xx(i+1,j+1,k) + sh_xx(i,j,k)) &
                                 + (sh_xx(i+1,j,k) + sh_xx(i,j+1,k)))
    enddo ; enddo

    ! compute Txx, Tyy tensor
    do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
      if (CS%ZB_type .NE. 1) then
        sum_sq = 0.5 * &
          (vort_xy_center(i,j)**2 + sh_xy_center(i,j)**2 + sh_xx(i,j,k)**2)
      endif

      if (CS%ZB_type .NE. 2) then
        vort_sh = vort_xy_center(i,j) * sh_xy_center(i,j)
      endif
      
      Txx(i,j,k) = CS%kappa_h(i,j) * (- vort_sh + sum_sq)
      Tyy(i,j,k) = CS%kappa_h(i,j) * (+ vort_sh + sum_sq)
    enddo ; enddo

    ! Here we assume that Txy is initialized to zero
    if (CS%ZB_type .NE. 2) then
      do J=Jsq-1,Jeq ; do I=Isq-1,Ieq
        Txy(I,J,k) = CS%kappa_q(i,j) * vort_xy(I,J,k) * sh_xx_corner(I,J)
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
subroutine compute_stress_divergence(Txx, Tyy, Txy, h, fx, fy, G, GV, CS, &
                                     dx2h, dy2h, dx2q, dy2q)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure
  type(ZB2020_CS),         intent(in) :: CS   !< ZB2020 control structure.

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),     &
        intent(inout) :: Txx,     & !< Subgrid stress xx component in h [L2 T-2 ~> m2 s-2]
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

  real, dimension(SZI_(G),SZJ_(G)),           & 
        intent(in) :: dx2q, &       !< dx^2 at q points [L2 ~> m2]
                      dy2q          !< dy^2 at q points [L2 ~> m2]

  real :: h_neglect    ! Thickness so small it can be lost in 
                       ! roundoff and so neglected [H ~> m or kg m-2]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  call cpu_clock_begin(CS%id_clock_divergence)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  h_neglect  = GV%H_subroundoff ! Line 410 on MOM_hor_visc.F90

  fx(:,:,:) = 0.
  fy(:,:,:) = 0.

  do k=1,nz
    ! Attenuation of the stress
    ! According to Klower
    if (CS%Klower_R_diss > 0.) then
      do J=Jsq,Jeq+1 ; do i=Isq,Ieq+1
        Txx(i,j,k) = Txx(i,j,k) * CS%c_diss(i,j,k)
        Tyy(i,j,k) = Tyy(i,j,k) * CS%c_diss(i,j,k)
      enddo ; enddo

      do J=js-1,Jeq ; do I=is-1,Ieq
        Txy(I,J,k) = Txy(I,J,k) * &
        0.25 * ((CS%c_diss(I,J  ,k) + CS%c_diss(I+1,J+1,k)) &
              + (CS%c_diss(I,J+1,k) + CS%c_diss(I+1,J  ,k)))
      enddo ; enddo
    endif

    ! Accounting for the varying thickness
    do J=Jsq,Jeq+1 ; do i=Isq,Ieq+1
      Txx(i,j,k) = Txx(i,j,k) * h(i,j,k)
      Tyy(i,j,k) = Tyy(i,j,k) * h(i,j,k)
    enddo ; enddo

    do J=js-1,Jeq ; do I=is-1,Ieq
      Txy(I,J,k) = Txy(I,J,k) * (CS%hq(I,J,k) * G%mask2dBu(I,J))
    enddo ; enddo

    ! Evaluate 1/h x.Div(h S) (Line 1495 of MOM_hor_visc.F90)
    ! Minus occurs because in original file (du/dt) = - div(S),
    ! but here is the discretization of div(S)
    do j=js,je ; do I=Isq,Ieq
      fx(I,j,k) = - ((G%IdyCu(I,j)*(dy2h(i,j)  *Txx(i,j,k)    - &
                                    dy2h(i+1,j)*Txx(i+1,j,k)) + &
                      G%IdxCu(I,j)*(dx2q(I,J-1)*Txy(I,J-1,k)  - &
                                    dx2q(I,J)  *Txy(I,J,k)))  * &
                      G%IareaCu(I,j)) / (CS%h_u(I,j,k) + h_neglect)
    enddo ; enddo

    ! Evaluate 1/h y.Div(h S) (Line 1517 of MOM_hor_visc.F90)
    do J=Jsq,Jeq ; do i=is,ie
      fy(i,J,k) = - ((G%IdyCv(i,J)*(dy2q(I-1,J)*Txy(I-1,J,k)   - &
                                    dy2q(I,J)  *Txy(I,J,k))    + & ! NOTE this plus
                      G%IdxCv(i,J)*(dx2h(i,j)  *Tyy(i,j,k)     - &
                                    dx2h(i,j+1)*Tyy(i,j+1,k))) * &
                      G%IareaCv(i,J)) / (CS%h_v(i,J,k) + h_neglect)
    enddo ; enddo

  enddo ! end of k loop

  call cpu_clock_end(CS%id_clock_divergence)

end subroutine compute_stress_divergence

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