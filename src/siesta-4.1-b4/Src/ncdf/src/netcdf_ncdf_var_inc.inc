#include "settings.inc"
subroutine ROUTINE3(put_var,VAR,name)(this,name,var,start,count)
  type(hNCDF), intent(inout) :: this
  character(len=*), intent(in) :: name
  VAR_TYPE, intent(in)DIMS :: var
  integer, intent(in), optional :: start(:)
  integer, intent(in), optional :: count(:)
  integer :: id
#ifdef IS_COMPLEX
  REAL_TYPE, allocatable DIMS :: r
#endif

#if DIM == 0
  call ncdf_put_var(this,name,(/var/),start=start,count=count)
#else

  if ( .not. ncdf_participate(this) ) return
  if ( this%define > -1 ) call ncdf_enddef(this)
#ifdef IS_COMPLEX
  ALLOC(r,var)
  r = real(var)
  call ncdf_inq_var(this,'Re'/ /name,id=id)
  call ncdf_err(nf90_put_var(this%id, id, r,start=start,count=count), &
     "Saving variable (VAR) Re"/ /trim(name)/ /' in file: '/ /this)
  call ncdf_inq_var(this,'Im'/ /name,id=id)
  r = aimag(var)
  call ncdf_err(nf90_put_var(this%id, id, r,start=start,count=count), &
       'Saving variable (VAR) Im'/ /trim(name)/ /' in file: '/ /this)
  deallocate(r)
#else
  call ncdf_inq_var(this,name,id=id)
  call ncdf_err(nf90_put_var(this%id, id, var,start=start,count=count), &
       'Saving variable (VAR) '/ /trim(name)/ /' in file: '/ /this)
#endif
#endif
end subroutine ROUTINE3(put_var,VAR,name)


subroutine ROUTINE3(get_var,VAR,name)(this, name, var, start, count, stride)
  type(hNCDF), intent(inout) :: this
  character(len=*), intent(in) :: name
  VAR_TYPE, intent(out)DIMS :: var
  integer, intent(in), optional :: start(:)
  integer, intent(in), optional :: count(:)
  integer, intent(in), optional :: stride(:)
#if DIM > 0 
  integer :: id
#ifdef IS_COMPLEX
  REAL_TYPE, allocatable DIMS :: r,i
#endif
#endif
#if DIM == 0
  VAR_TYPE :: v(1)
  call ncdf_get_var(this,name,v,start=start,count=count,stride=stride)
  var = v(1)
#else
  if ( .not. ncdf_participate(this) ) return
  if ( this%define > -1 ) call ncdf_enddef(this)
#ifdef IS_COMPLEX
  ALLOC(r,var)
  ALLOC(i,var)
  call ncdf_inq_var(this,'Re'/ /name,id=id)
  call ncdf_err(nf90_get_var(this%id, id, r,start=start,count=count,stride=stride), &
       'Retrieving variable (VAR) Re'/ /trim(name)/ /' in file: '/ /this)
  call ncdf_inq_var(this,'Im'/ /name,id=id)
  call ncdf_err(nf90_get_var(this%id, id, i,start=start,count=count,stride=stride), &
       'Retrieving variable (VAR) Im'/ /trim(name)/ /' in file: '/ /this)
  var = cmplx(r,i)
  deallocate(r,i)
#else
  call ncdf_inq_var(this,name,id=id)
  call ncdf_err(nf90_get_var(this%id, id, var,start=start,count=count,stride=stride), &
       'Retrieving (VAR) variable, '/ /trim(name)/ /' in file: '/ /this)
#endif
#endif
end subroutine ROUTINE3(get_var,VAR,name)


#ifdef VAR_FILL_IN_FORTRAN_NOT_IMPLEMENTED_YET
# ! we need to put back the tmp_lr to fill_val
#else

#if DIM == 0
subroutine ROUTINE2(def_fill,VAR)(this, name, fill_val, fill)
  type(hNCDF), intent(inout) :: this
  character(len=*), intent(in) :: name
  VAR_TYPE, intent(in) :: fill_val ! non-optional to allow interfacing
  integer, intent(in), optional :: fill
  integer :: lfill
  VAR_TYPE :: lfill_val	
  integer :: tmp_lr
#ifdef IS_COMPLEX
  REAL_TYPE :: lr
  integer :: id(2)
#else
  VAR_TYPE :: lr
  integer :: id
#endif
  if ( .not. ncdf_participate(this) ) return

  call ncdf_redef(this)

  call ncdf_inq_var(this,name,id=id,fill=lfill,fill_val=lfill_val)
  if ( present(fill) ) lfill = fill
  lfill_val = fill_val

#ifdef IS_COMPLEX
  lr = real(lfill_val)
  call ncdf_err(nf90_def_var_fill(this%id, id(1), lfill, tmp_lr), &
       'Setting fill (VAR) Re'/ /trim(name)/ /' in file: '/ /this)
  lr = aimag(lfill_val)
  call ncdf_err(nf90_def_var_fill(this%id, id(2), lfill, tmp_lr), &
       'Setting fill (VAR) Im'/ /trim(name)/ /' in file: '/ /this)
#else
  call ncdf_err(nf90_def_var_fill(this%id, id, lfill, tmp_lr), &
       'Setting fill (VAR) variable, '/ /trim(name)/ /' in file: '/ /this)
#endif
end subroutine ROUTINE2(def_fill,VAR)

subroutine ROUTINE2(inq_var,VAR)(this, name,fill_val,exist,id,size,atts,fill)
  use dictionary
  type(hNCDF), intent(inout) :: this
  character(len=*), intent(in) :: name
  VAR_TYPE, intent(out) :: fill_val ! non-optional to allow interfacing
  logical, intent(out), optional  :: exist
#ifdef IS_COMPLEX
  integer, intent(out), optional  :: id(2)
#else
  integer, intent(out), optional  :: id
#endif
  integer, intent(out), optional  :: size(:)
  type(dict), intent(inout), optional :: atts
  integer, intent(out), optional :: fill
  integer :: lid, lfill
  integer :: tmp_lr
#ifdef IS_COMPLEX
  REAL_TYPE :: lfill_valr,lfill_valc
#else
  REAL_TYPE :: lfill_val
#endif
  if ( .not. ncdf_participate(this) ) return
#ifdef IS_COMPLEX
  call ncdf_inq_var_def(this,'Re'/ /name,id=lid,exist=exist,size=size,atts=atts)
  if ( present(exist) ) then
     if ( .not. exist ) return
  end if
  if ( present(id) ) id(1) = lid
  call ncdf_err(nf90_inq_var_fill(this%id, lid, lfill, tmp_lr), &
       'Retrieving variable-fill (VAR) Re-'/ /trim(name)/ /' in file: '/ /this)
  if ( present(fill) ) fill = lfill
  call ncdf_inq_var_def(this,'Im'/ /name,id=lid,size=size,atts=atts)
  if ( present(id) ) id(2) = lid
  call ncdf_err(nf90_inq_var_fill(this%id, lid, lfill, tmp_lr), &
       'Retrieving variable-fill (VAR) Im-'/ /trim(name)/ /' in file: '/ /this)
  if ( present(fill) ) then
     if ( fill /= lfill ) then	
     	call ncdf_err(-100,'Fill-value for real and imaginary part'/ /&
	  ' are not the same. This is not allowed.')
     end if
  end if
!  if ( present(fill_val) ) then
     !fill_val = cmplx(lfill_valr,lfill_valc)
     fill_val = cmplx(0.,0.)
!  end if
#else
  call ncdf_inq_var_def(this,name,id=lid,exist=exist,size=size,atts=atts)
  if ( present(exist) ) then
     if ( .not. exist ) return
  end if
  if ( present(id) ) id = lid
  call ncdf_err(nf90_inq_var_fill(this%id, lid, lfill, tmp_lr), &
       'Retrieving variable-fill (VAR) '/ /trim(name)/ /' in file: '/ /this)
  if ( present(fill) ) fill = lfill
  !if ( present(fill_val) ) fill_val = lfill_val
  fill_val = 0.
#endif
end subroutine ROUTINE2(inq_var,VAR)

#endif

#endif

#undef __NC_COUNT
