! Module for regular interpolation...

! Fully created by Nick Papior Andersen


module m_interpolate

  ! Generic module for interpolation
  integer, parameter :: dp = selected_real_kind(14,100)

  private

  public :: crt_pivot
  public :: interp_linear
  public :: interp_spline
  public :: prep_spline

contains

  subroutine crt_pivot(N,x,ipvt)
    integer, intent(in) :: N
    real(dp), intent(in) :: x(N)
    integer, intent(out) :: ipvt(N)
    integer :: i, j, ip, i_cur
    real(dp) :: x_cur, xd
    
    ! Find minimum x
    x_cur = minval(x)
    i_cur = minloc(x,dim=1)
    ip = 1
    ipvt(1) = i_cur
    do while ( ip < N )
       ! Ensure compactness
       xd = huge(1._dp)
       do i = 1 , N
          if ( i_cur == i ) cycle ! current point
          if ( any(i == ipvt(1:ip)) ) cycle ! already found
          if ( x(i) - x_cur < xd ) then
             ! save possible current place
             j = i
             xd = x(i) - x_cur
          end if
       end do

       ! Save found position
       ip = ip + 1 
       ipvt(ip) = j
       x_cur = x(j)

    end do

  end subroutine crt_pivot


  ! A simple linear interpolation algorithm
  subroutine interp_linear(N,x,y,x0,y0)
    integer, intent(in) :: N
    real(dp), intent(in) :: x(N), y(N)
    real(dp), intent(in) :: x0
    real(dp), intent(out) :: y0
    
    ! Local variables
    integer :: i, ipvt(N)
    real(dp) :: b

    if ( N < 1 ) then
       y0 = 0._dp
       return
    end if

    if ( N == 1 ) then
       y0 = y(1)
       return
    end if

    ! Create pivoting array for correct sorting
    ! in case the user did not provide a consecutive
    ! ordering...
    call crt_pivot(N,x,ipvt)

    ! Do extrapolation
    !  1. lower x
    !  2. higher x
    if ( x0 <= x(ipvt(1)) ) then
       b  = (y(ipvt(2)) - y(ipvt(1)))/(x(ipvt(2)) - x(ipvt(1)))
       y0 = y(ipvt(1)) + b * (x0 - x(ipvt(1)))
       return
    else if ( x(ipvt(N)) < x0 ) then
       b  = (y(ipvt(N)) - y(ipvt(N-1)))/(x(ipvt(N)) - x(ipvt(N-1)))
       y0 = y(ipvt(N)) + b * (x0 - x(ipvt(N)))
       return
    end if

    do i = 1 , N - 1
       if ( x0 <= x(ipvt(i+1)) ) then
          b  = (y(ipvt(i+1)) - y(ipvt(i)))/(x(ipvt(i+1)) - x(ipvt(i)))
          y0 = y(ipvt(i)) + b * (x0 - x(ipvt(i)))
          return
       end if
    end do
          
  end subroutine interp_linear

  subroutine interp_spline(N,x,y,x0,y0,z)
    integer, intent(in) :: N
    real(dp), intent(in) :: x(N), y(N)
    real(dp), intent(in) :: x0
    real(dp), intent(out) :: y0
    real(dp), intent(in), optional :: z(N-2) ! z as calculated by prep_spline
    
    ! Local variables
    integer :: i, ipvt(N), idx, seg
    real(dp) :: b
    real(dp), allocatable :: zz(:)

    if ( N < 1 ) then
       y0 = 0._dp
       return
    end if

    ! If we have 2 or less points we do linear interp
    if ( N <= 2 ) then
       call interp_linear(N,x,y,x0,y0)
       return
    end if

    ! Create pivoting array for correct sorting
    ! in case the user did not provide a consecutive
    ! ordering...
    call crt_pivot(N,x,ipvt)

    ! If we are out-of-bounds we do extrapolation (from a linear
    ! perspective)
    if ( x0 <= x(ipvt(1)) ) then
       b = (y(ipvt(2)) - y(ipvt(1)))/(x(ipvt(2)) - x(ipvt(1)))
       y0 = y(ipvt(1)) + b * (x0 - x(ipvt(1)))
       return
    else if ( x(ipvt(N)) <= x0 ) then
       b = (y(ipvt(N)) - y(ipvt(N-1)))/(x(ipvt(N)) - x(ipvt(N-1)))
       y0 = y(ipvt(N)) + b * (x0 - x(ipvt(N)))
       return
    end if

    if ( present(z) ) then
       ! We calculate from already calculated parameters...
       do i = 1 , N - 1 ! z(1:N-2)
          if ( x0 <= x(ipvt(i+1)) ) then
             idx = i
             exit
          end if
       end do
       if ( idx == 1 ) then
          seg = -1
       else if ( idx == N - 1 ) then
          seg = 1
       else
          seg = 0
       end if
       i = idx
       call interp(seg,x(ipvt(i)), y(ipvt(i)), &
            x(ipvt(i+1)), y(ipvt(i+1)), z(max(i-1,1)), x0, y0)
    else

       ! We now need to calculate it manually...
       do i = 1 , N - 1 ! z(1:N-1)
          if ( x0 <= x(ipvt(i+1)) ) then
             ! We have found the position
             idx = i
             exit
          end if
       end do

       ! Allocate zz 
       i = N - idx
       allocate(zz(max(i,2)))
       zz(1:2) = 0._dp

       ! only calculate the necessary z's
       call loc_prep_spline(N,x,y,ipvt,max(i,2),zz)

       if ( idx == 1 ) then
          zz(2) = zz(1)
          zz(1) = 0._dp
       else if ( idx == N - 1 ) then
          zz(1) = zz(2)
          zz(2) = 0._dp
       end if
       i = idx
       call interp(0,x(ipvt(i)), y(ipvt(i)), &
            x(ipvt(i+1)), y(ipvt(i+1)), zz(1), x0, y0)

       deallocate(zz)

    end if

  contains

    subroutine interp(seg,x0,y0,x1,y1,z,x,y)
      integer, intent(in) :: seg ! -1 (start), 0 (middle), 1 (end)
      real(dp), intent(in) :: x0, y0, x1, y1, z(2), x
      real(dp), intent(out) :: y
      real(dp) :: h, xd, b

      ! We have found the point...
      h  = x1 - x0
      xd = x  - x0
         
      if ( seg == 1 ) then
         ! third order polynomial
         y = - z(1) / ( 6._dp * h ) * xd
         ! second order polynomial
         y = ( y + z(1) * .5_dp ) * xd
         ! first order polynomial
         b = - h / 3._dp * z(1) + ( y1 - y0 ) / h
         y = ( y + b ) * xd
      else if ( seg == 0 ) then
         ! third order polynomial
         y = ( z(2) - z(1) ) / ( 6._dp * h ) * xd
         ! second order polynomial
         y = ( y + z(1) * .5_dp ) * xd
         ! first order polynomial
         b = - h / 6._dp * z(2)
         b = b - h / 3._dp * z(1) + ( y1 - y0 ) / h
         y = ( y + b ) * xd
      else if ( seg == -1 ) then
         ! third order polynomial
         y = z(1) / ( 6._dp * h ) * xd
         ! second order polynomial
         y = y * xd
         ! first order polynomial
         b = - h / 6._dp * z(1)
         b = b + ( y1 - y0 ) / h
         y = ( y + b ) * xd
      end if
      ! zeroth order
      y = y + y0
         
    end subroutine interp

  end subroutine interp_spline

  subroutine loc_prep_spline(N,x,y,ipvt,NZ,z)
    integer, intent(in) :: N
    real(dp), intent(in) :: x(N), y(N)
    integer, intent(in) :: ipvt(N)
    integer, intent(in):: NZ
    real(dp), intent(inout) :: z(NZ)

    ! Local variables
    integer :: i, iz
    real(dp) :: u(N-2), v(N-2)
    real(dp) :: b(2), h(2)

    ! Calculate u_i/v_i
    h(1) = x(ipvt(2)) - x(ipvt(1))
    b(1) = (y(ipvt(2)) - y(ipvt(1))) / h(1)
    h(2) = x(ipvt(3)) - x(ipvt(2))
    b(2) = (y(ipvt(3)) - y(ipvt(2))) / h(2)
    u(1) = 2._dp * ( h(1) + h(2) )
    v(1) = 6._dp * ( b(2) - b(1) )
    do i = 2 , N - 2
       ! Transfer h,b
       h(1) = h(2)
       b(1) = b(2)
       h(2) = x(ipvt(i+2)) - x(ipvt(i+1))
       b(2) = ( y(ipvt(i+2)) - y(ipvt(i+1)) ) / h(2)

       ! Calculate u_i,v_i
       u(i) = 2._dp * ( h(1) + h(2) ) - h(1) ** 2 / u(i-1)
       v(i) = 6._dp * ( b(2) - b(1) ) - h(1) * v(i-1) / u(i-1)

    end do

    ! for the case of a limited number of z's, we simply
    ! calculate them like this:
    ! Calculate z_i
    iz = min(NZ,N-2)
    z(iz) = v(N-2) / u(N-2)
    do i = N - 3 , 1 , -1
       iz = iz - 1
       if ( iz == 0 ) return
       h(1) = x(ipvt(i+2)) - x(ipvt(i+1))
       z(iz) = ( v(i) - h(1) * z(iz+1) ) / u(i)
    end do

  end subroutine loc_prep_spline
    

  subroutine prep_spline(N,x,y,z,NZ)
    integer, intent(in) :: N
    real(dp), intent(in) :: x(N), y(N)
    real(dp), intent(out) :: z(:)
    integer, intent(in), optional :: NZ
    
    ! Local variables
    integer :: ipvt(N)

    if ( N <= 2 ) then
       ! We cannot do anything...
       return
    end if

    ! Create pivoting array for correct sorting
    ! in case the user did not provide a consecutive
    ! ordering...
    call crt_pivot(N,x,ipvt)
    
    if ( present(NZ) ) then
       call loc_prep_spline(N,x,y,ipvt,NZ,z)
    else
       call loc_prep_spline(N,x,y,ipvt,N-2,z)
    end if

  end subroutine prep_spline

end module m_interpolate



