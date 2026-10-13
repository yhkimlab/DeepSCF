      MODULE MPI__r4_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r4_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(4), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(4), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         real(4), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         real(4), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(4), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       real(4), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r4_V
      MODULE MPI__c4_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c4_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(4), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(4), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         complex(4), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         complex(4), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(4), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       complex(4), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c4_V
      MODULE MPI__r8_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r8_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(8), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(8), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         real(8), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         real(8), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(8), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       real(8), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r8_V
      MODULE MPI__c8_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c8_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(8), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(8), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         complex(8), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         complex(8), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(8), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       complex(8), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c8_V
      MODULE MPI__i4_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i4_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(4), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(4), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         integer(4), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         integer(4), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(4), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       integer(4), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i4_V
      MODULE MPI__i8_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i8_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(8), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(8), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         integer(8), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         integer(8), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(8), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       integer(8), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i8_V
      MODULE MPI__logical_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__logical_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          logical, INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          logical, INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          logical, INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         logical, INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         logical, INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          logical, INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       logical, INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          logical, INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          logical, INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          logical, INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__logical_V
      MODULE MPI__character_V
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__character_V
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

        PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

        PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!       PUBLIC :: MPI_SENDRECV
!       INTERFACE MPI_SENDRECV
!         MODULE PROCEDURE MPI_SENDRECV_T
!       END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          character(*), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          character(*), INTENT(IN)  :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          character(*), INTENT(OUT) :: BUFFER(*) 
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(OUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         character(*), INTENT(IN)  :: SENDBUF(*)
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         character(*), INTENT(OUT) :: RECVBUF(*) 
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          character(*), INTENT(INOUT) :: BUF(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       character(*), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          character(*), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          character(*), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(INOUT) :: BUFFER(*) 
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__character_V
      MODULE MPI__r4_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r4_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(4), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(4), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         real(4), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         real(4), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(4), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       real(4), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r4_S
      MODULE MPI__c4_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c4_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(4), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(4), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         complex(4), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         complex(4), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(4), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       complex(4), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c4_S
      MODULE MPI__r8_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r8_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(8), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          real(8), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          real(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         real(8), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         real(8), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          real(8), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       real(8), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r8_S
      MODULE MPI__c8_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c8_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(8), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          complex(8), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          complex(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         complex(8), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         complex(8), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          complex(8), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       complex(8), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c8_S
      MODULE MPI__i4_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i4_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(4), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(4), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(4), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         integer(4), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         integer(4), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(4), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       integer(4), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i4_S
      MODULE MPI__i8_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i8_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(8), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          integer(8), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          integer(8), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         integer(8), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         integer(8), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          integer(8), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       integer(8), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i8_S
      MODULE MPI__logical_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__logical_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          logical, INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          logical, INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          logical, INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          logical, INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         logical, INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         logical, INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          logical, INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       logical, INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          logical, INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          logical, INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          logical, INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__logical_S
      MODULE MPI__character_S
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__character_S
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        PUBLIC :: MPI_SEND
        INTERFACE MPI_SEND
          MODULE PROCEDURE MPI_SEND_T
        END INTERFACE ! MPI_SEND

        PUBLIC :: MPI_RECV
        INTERFACE MPI_RECV
          MODULE PROCEDURE MPI_RECV_T
        END INTERFACE ! MPI_RECV

        PUBLIC :: MPI_BSEND
        INTERFACE MPI_BSEND
          MODULE PROCEDURE MPI_BSEND_T
        END INTERFACE ! MPI_BSEND

        PUBLIC :: MPI_SSEND
        INTERFACE MPI_SSEND
          MODULE PROCEDURE MPI_SSEND_T
        END INTERFACE ! MPI_SSEND

        PUBLIC :: MPI_RSEND
        INTERFACE MPI_RSEND
          MODULE PROCEDURE MPI_RSEND_T
        END INTERFACE ! MPI_RSEND

!!!     PUBLIC :: MPI_BUFFER_ATTACH
        INTERFACE MPI_BUFFER_ATTACH
          MODULE PROCEDURE MPI_BUFFER_ATTACH_T
        END INTERFACE ! MPI_BUFFER_ATTACH

!!!     PUBLIC :: MPI_BUFFER_DETACH
        INTERFACE MPI_BUFFER_DETACH
          MODULE PROCEDURE MPI_BUFFER_DETACH_T
        END INTERFACE ! MPI_BUFFER_DETACH

        PUBLIC :: MPI_ISEND
        INTERFACE MPI_ISEND
          MODULE PROCEDURE MPI_ISEND_T
        END INTERFACE ! MPI_ISEND

        PUBLIC :: MPI_IBSEND
        INTERFACE MPI_IBSEND
          MODULE PROCEDURE MPI_IBSEND_T
        END INTERFACE ! MPI_IBSEND

        PUBLIC :: MPI_ISSEND
        INTERFACE MPI_ISSEND
          MODULE PROCEDURE MPI_ISSEND_T
        END INTERFACE ! MPI_ISSEND

        PUBLIC :: MPI_IRSEND
        INTERFACE MPI_IRSEND
          MODULE PROCEDURE MPI_IRSEND_T
        END INTERFACE ! MPI_IRSEND

        PUBLIC :: MPI_IRECV
        INTERFACE MPI_IRECV
          MODULE PROCEDURE MPI_IRECV_T
        END INTERFACE ! MPI_IRECV

        PUBLIC :: MPI_SEND_INIT
        INTERFACE MPI_SEND_INIT
          MODULE PROCEDURE MPI_SEND_INIT_T
        END INTERFACE ! MPI_SEND_INIT

        PUBLIC :: MPI_BSEND_INIT
        INTERFACE MPI_BSEND_INIT
          MODULE PROCEDURE MPI_BSEND_INIT_T
        END INTERFACE ! MPI_BSEND_INIT

        PUBLIC :: MPI_SSEND_INIT
        INTERFACE MPI_SSEND_INIT
          MODULE PROCEDURE MPI_SSEND_INIT_T
        END INTERFACE ! MPI_SSEND_INIT

        PUBLIC :: MPI_RSEND_INIT
        INTERFACE MPI_RSEND_INIT
          MODULE PROCEDURE MPI_RSEND_INIT_T
        END INTERFACE ! MPI_RSEND_INIT

        PUBLIC :: MPI_RECV_INIT
        INTERFACE MPI_RECV_INIT
          MODULE PROCEDURE MPI_RECV_INIT_T
        END INTERFACE ! MPI_RECV_INIT

!!!     PUBLIC :: MPI_SENDRECV
!!!     INTERFACE MPI_SENDRECV
!!!       MODULE PROCEDURE MPI_SENDRECV_T
!!!     END INTERFACE ! MPI_SENDRECV

        PUBLIC :: MPI_SENDRECV_REPLACE
        INTERFACE MPI_SENDRECV_REPLACE
          MODULE PROCEDURE MPI_SENDRECV_REPLACE_T
        END INTERFACE ! MPI_SENDRECV_REPLACE

!!!     PUBLIC :: MPI_ADDRESS
!!!     INTERFACE MPI_ADDRESS
!!!       MODULE PROCEDURE MPI_ADDRESS_T
!!!     END INTERFACE ! MPI_ADDRESS

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_BCAST
        INTERFACE MPI_BCAST
          MODULE PROCEDURE MPI_BCAST_T
        END INTERFACE ! MPI_BCAST

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_SEND_T(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND
          call timer_mpi('MPI_SEND',1)
          CALL     MPI_SEND(                                            &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          call timer_mpi('MPI_SEND',2)
        END SUBROUTINE MPI_SEND_T
        
        SUBROUTINE MPI_RECV_T(                                          &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          character(*), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV
          call timer_mpi('MPI_RECV',1)
          CALL     MPI_RECV(                                            &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR)
          call timer_mpi('MPI_RECV',2)
        END SUBROUTINE MPI_RECV_T
        
        SUBROUTINE MPI_BSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND
          CALL     MPI_BSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_BSEND_T
        
        SUBROUTINE MPI_SSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND
          CALL     MPI_SSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_SSEND_T
        
        SUBROUTINE MPI_RSEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND
          CALL     MPI_RSEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR)
        END SUBROUTINE MPI_RSEND_T
        
        SUBROUTINE MPI_BUFFER_ATTACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          character(*), INTENT(IN)  :: BUFFER(*)
          INTEGER, INTENT(IN)  :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_ATTACH
          CALL     MPI_BUFFER_ATTACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_ATTACH_T
        
        SUBROUTINE MPI_BUFFER_DETACH_T(                                 &
     &      BUFFER, SIZE, IERROR)
          character(*), INTENT(OUT) :: BUFFER(*)
          INTEGER, INTENT(OUT) :: SIZE
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BUFFER_DETACH
          CALL     MPI_BUFFER_DETACH(                                   &
     &      BUFFER, SIZE, IERROR)
        END SUBROUTINE MPI_BUFFER_DETACH_T
        
        SUBROUTINE MPI_ISEND_T(                                         &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISEND
          call timer_mpi('MPI_ISEND',1)
          CALL     MPI_ISEND(                                           &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_ISEND',2)
        END SUBROUTINE MPI_ISEND_T
        
        SUBROUTINE MPI_IBSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IBSEND
          CALL     MPI_IBSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IBSEND_T
        
        SUBROUTINE MPI_ISSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ISSEND
          CALL     MPI_ISSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_ISSEND_T
        
        SUBROUTINE MPI_IRSEND_T(                                        &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRSEND
          CALL     MPI_IRSEND(                                          &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_IRSEND_T
        
        SUBROUTINE MPI_IRECV_T(                                         &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_IRECV
          call timer_mpi('MPI_IRECV',1)
          CALL     MPI_IRECV(                                           &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          call timer_mpi('MPI_IRECV',2)
        END SUBROUTINE MPI_IRECV_T
        
        SUBROUTINE MPI_SEND_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SEND_INIT
          CALL     MPI_SEND_INIT(                                       &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SEND_INIT_T
        
        SUBROUTINE MPI_BSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BSEND_INIT
          CALL     MPI_BSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_BSEND_INIT_T
        
        SUBROUTINE MPI_SSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SSEND_INIT
          CALL     MPI_SSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_SSEND_INIT_T
        
        SUBROUTINE MPI_RSEND_INIT_T(                                    &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(IN)  :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RSEND_INIT
          CALL     MPI_RSEND_INIT(                                      &
     &      BUF, COUNT, DATATYPE, DEST, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RSEND_INIT_T
        
        SUBROUTINE MPI_RECV_INIT_T(                                     &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
          character(*), INTENT(OUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: TAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: REQUEST
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_RECV_INIT
          CALL     MPI_RECV_INIT(                                       &
     &      BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, REQUEST, IERROR)
        END SUBROUTINE MPI_RECV_INIT_T
        
!       SUBROUTINE MPI_SENDRECV_T(                                      &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!         USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
!         character(*), INTENT(IN)  :: SENDBUF
!         INTEGER, INTENT(IN)  :: SENDCOUNT
!         INTEGER, INTENT(IN)  :: SENDTYPE
!         INTEGER, INTENT(IN)  :: DEST
!         INTEGER, INTENT(IN)  :: SENDTAG
!         character(*), INTENT(OUT) :: RECVBUF
!         INTEGER, INTENT(IN)  :: RECVCOUNT
!         INTEGER, INTENT(IN)  :: RECVTYPE
!         INTEGER, INTENT(IN)  :: SOURCE
!         INTEGER, INTENT(IN)  :: RECVTAG
!         INTEGER, INTENT(IN)  :: COMM
!         INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
!         INTEGER, INTENT(OUT) :: IERROR 
!         EXTERNAL MPI_SENDRECV
!         CALL     MPI_SENDRECV(                                        &
!    &      SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF,       &
!    &      RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR)
!       END SUBROUTINE MPI_SENDRECV_T
        
        SUBROUTINE MPI_SENDRECV_REPLACE_T(                              &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
          USE MPI__INCLUDE, ONLY: MPI_STATUS_SIZE
          character(*), INTENT(INOUT) :: BUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: DEST
          INTEGER, INTENT(IN)  :: SENDTAG
          INTEGER, INTENT(IN)  :: SOURCE
          INTEGER, INTENT(IN)  :: RECVTAG
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: STATUS(MPI_STATUS_SIZE)
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SENDRECV_REPLACE
          CALL     MPI_SENDRECV_REPLACE(                                &
     &      BUF, COUNT, DATATYPE, DEST, SENDTAG, SOURCE, RECVTAG,       &
     &      COMM, STATUS, IERROR)
        END SUBROUTINE MPI_SENDRECV_REPLACE_T
        
!!!     SUBROUTINE MPI_ADDRESS_T(                                       &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!       character(*), INTENT(IN)  :: LOCATION(*) 
!!!       INTEGER, INTENT(OUT) :: ADDRESS
!!!       INTEGER, INTENT(OUT) :: IERROR 
!!!       EXTERNAL MPI_ADDRESS
!!!       CALL     MPI_ADDRESS(                                         &
!!!  &      LOCATION, ADDRESS, IERROR)
!!!     END SUBROUTINE MPI_ADDRESS_T
        
        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          character(*), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          character(*), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_BCAST_T(                                         &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(INOUT) :: BUFFER
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_BCAST
          call timer_mpi('MPI_BCAST',1)
          CALL     MPI_BCAST(                                           &
     &      BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_BCAST',2)
        END SUBROUTINE  MPI_BCAST_T
                  
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__character_S
      MODULE MPI__r4_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r4_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF
          real(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r4_SV
      MODULE MPI__c4_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c4_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF
          complex(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c4_SV
      MODULE MPI__r8_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r8_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF
          real(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r8_SV
      MODULE MPI__c8_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c8_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF
          complex(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c8_SV
      MODULE MPI__i4_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i4_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(4), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF
          integer(4), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i4_SV
      MODULE MPI__i8_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i8_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(8), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF
          integer(8), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i8_SV
      MODULE MPI__logical_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__logical_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          logical, INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          logical, INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF
          logical, INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__logical_SV
      MODULE MPI__character_SV
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__character_SV
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          character(*), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          character(*), INTENT(OUT) :: OUTBUF(*)
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF
          character(*), INTENT(OUT) :: RECVBUF(*)
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__character_SV
      MODULE MPI__r4_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r4_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(4), INTENT(IN)  :: SENDBUF(*)
          real(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r4_VS
      MODULE MPI__c4_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c4_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(4), INTENT(IN)  :: SENDBUF(*)
          complex(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c4_VS
      MODULE MPI__r8_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__r8_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          real(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          real(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          real(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          real(8), INTENT(IN)  :: SENDBUF(*)
          real(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__r8_VS
      MODULE MPI__c8_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__c8_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          complex(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          complex(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          complex(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          complex(8), INTENT(IN)  :: SENDBUF(*)
          complex(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__c8_VS
      MODULE MPI__i4_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i4_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(4), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(4), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(4), INTENT(IN)  :: SENDBUF(*)
          integer(4), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i4_VS
      MODULE MPI__i8_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__i8_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          integer(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          integer(8), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          integer(8), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          integer(8), INTENT(IN)  :: SENDBUF(*)
          integer(8), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__i8_VS
      MODULE MPI__logical_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__logical_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          logical, INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          logical, INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          logical, INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          logical, INTENT(IN)  :: SENDBUF(*)
          logical, INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__logical_VS
      MODULE MPI__character_VS
!
!
!       Michael Hennecke
!       A Fortran 90 interface to MPI version 1.1
!       RZ Uni Karlsruhe, Internal Report 63/96
!
!         Auxiliary module template MPI__character_VS
!
!         03-Oct-1996, hennecke@rz.uni-karlsruhe.de (v0.9c beta)
!
!       Permission is granted to copy and distribute this file
!       or modified versions of this file for no fee, provided the 
!       copyright notice and this permission notice are preserved 
!       on all copies.
!
!       (C) 1996  Michael Hennecke, RZ Universitaet Karlsruhe
!
!
        USE TIMER_MPI_M, only: timer_mpi

        IMPLICIT NONE
        PRIVATE

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

!!!     PUBLIC :: MPI_PACK
!!!     INTERFACE MPI_PACK
!!!       MODULE PROCEDURE MPI_PACK_T
!!!     END INTERFACE ! MPI_PACK

!!!     PUBLIC :: MPI_UNPACK
!!!     INTERFACE MPI_UNPACK
!!!       MODULE PROCEDURE MPI_UNPACK_T
!!!     END INTERFACE ! MPI_UNPACK

!       ... A.10  Fortran Bindings for Collective Communication  ...

        PUBLIC :: MPI_GATHER
        INTERFACE MPI_GATHER
          MODULE PROCEDURE MPI_GATHER_T
        END INTERFACE ! MPI_GATHER

        PUBLIC :: MPI_GATHERV
        INTERFACE MPI_GATHERV
          MODULE PROCEDURE MPI_GATHERV_T
        END INTERFACE ! MPI_GATHERV

        PUBLIC :: MPI_SCATTER
        INTERFACE MPI_SCATTER
          MODULE PROCEDURE MPI_SCATTER_T
        END INTERFACE ! MPI_SCATTER

        PUBLIC :: MPI_SCATTERV
        INTERFACE MPI_SCATTERV
          MODULE PROCEDURE MPI_SCATTERV_T
        END INTERFACE ! MPI_SCATTERV

        PUBLIC :: MPI_ALLGATHER
        INTERFACE MPI_ALLGATHER
          MODULE PROCEDURE MPI_ALLGATHER_T
        END INTERFACE ! MPI_ALLGATHER

        PUBLIC :: MPI_ALLGATHERV
        INTERFACE MPI_ALLGATHERV
          MODULE PROCEDURE MPI_ALLGATHERV_T
        END INTERFACE ! MPI_ALLGATHERV

        PUBLIC :: MPI_ALLTOALL
        INTERFACE MPI_ALLTOALL
          MODULE PROCEDURE MPI_ALLTOALL_T
        END INTERFACE ! MPI_ALLTOALL

        PUBLIC :: MPI_ALLTOALLV
        INTERFACE MPI_ALLTOALLV
          MODULE PROCEDURE MPI_ALLTOALLV_T
        END INTERFACE ! MPI_ALLTOALLV

        PUBLIC :: MPI_REDUCE
        INTERFACE MPI_REDUCE
          MODULE PROCEDURE MPI_REDUCE_T
        END INTERFACE ! MPI_REDUCE

        PUBLIC :: MPI_ALLREDUCE
        INTERFACE MPI_ALLREDUCE
          MODULE PROCEDURE MPI_ALLREDUCE_T
        END INTERFACE ! MPI_ALLREDUCE

        PUBLIC :: MPI_REDUCE_SCATTER
        INTERFACE MPI_REDUCE_SCATTER
          MODULE PROCEDURE MPI_REDUCE_SCATTER_T
        END INTERFACE ! MPI_REDUCE_SCATTER

        PUBLIC :: MPI_SCAN
        INTERFACE MPI_SCAN
          MODULE PROCEDURE MPI_SCAN_T
        END INTERFACE ! MPI_SCAN

      CONTAINS

!       ... A.9  Fortran Bindings for Point-to-Point Communication  ...

        SUBROUTINE MPI_PACK_T(                                          &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          character(*), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(INOUT) :: POSITION
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_PACK
          CALL     MPI_PACK(                                            &
     &      INBUF, INCOUNT, DATATYPE, OUTBUF, OUTCOUNT, POSITION,       &
     &      COMM, IERROR)
        END SUBROUTINE MPI_PACK_T
        
        SUBROUTINE MPI_UNPACK_T(                                        &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
          character(*), INTENT(IN)  :: INBUF(*)
          INTEGER, INTENT(IN)  :: INSIZE
          INTEGER, INTENT(INOUT) :: POSITION
          character(*), INTENT(OUT) :: OUTBUF
          INTEGER, INTENT(IN)  :: OUTCOUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_UNPACK
          CALL     MPI_UNPACK(                                          &
     &      INBUF, INSIZE, POSITION, OUTBUF, OUTCOUNT, DATATYPE,        &
     &      COMM, IERROR)
        END SUBROUTINE MPI_UNPACK_T
        
!      ... A.10  Fortran Bindings for Collective Communication  ...
        
        SUBROUTINE MPI_GATHER_T(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHER
          call timer_mpi('MPI_GATHER',1)
          CALL     MPI_GATHER(                                          &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_GATHER',2)
        END SUBROUTINE MPI_GATHER_T
        
        SUBROUTINE MPI_GATHERV_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_GATHERV
          CALL     MPI_GATHERV(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_GATHERV_T
                  
        SUBROUTINE MPI_SCATTER_T(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTER
          CALL     MPI_SCATTER(                                         &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTER_T
        
        SUBROUTINE MPI_SCATTERV_T(                                      &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCATTERV
          CALL     MPI_SCATTERV(                                        &
     &      SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF,             &
     &      RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) 
        END SUBROUTINE MPI_SCATTERV_T
        
        SUBROUTINE MPI_ALLGATHER_T(                                     &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHER
          call timer_mpi('MPI_ALLGATHER',1)
          CALL     MPI_ALLGATHER(                                       &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          call timer_mpi('MPI_ALLGATHER',2)
        END SUBROUTINE MPI_ALLGATHER_T
        
        SUBROUTINE MPI_ALLGATHERV_T(                                    &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLGATHERV
          CALL     MPI_ALLGATHERV(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS,          &
     &      DISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLGATHERV_T
        
        SUBROUTINE MPI_ALLTOALL_T(                                      &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNT
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNT
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALL
          CALL     MPI_ALLTOALL(                                        &
     &      SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT,           &
     &      RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALL_T
        
        SUBROUTINE MPI_ALLTOALLV_T(                                     &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          INTEGER, INTENT(IN)  :: SENDCOUNTS(*)
          INTEGER, INTENT(IN)  :: SDISPLS(*)
          INTEGER, INTENT(IN)  :: SENDTYPE
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: RDISPLS(*)
          INTEGER, INTENT(IN)  :: RECVTYPE
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLTOALLV
          CALL     MPI_ALLTOALLV(                                       &
     &      SENDBUF, SENDCOUNTS, SDISPLS, SENDTYPE, RECVBUF,            &
     &      RECVCOUNTS, RDISPLS, RECVTYPE, COMM, IERROR) 
        END SUBROUTINE MPI_ALLTOALLV_T
        
        SUBROUTINE MPI_REDUCE_T(                                        &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: ROOT
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE
          call timer_mpi('MPI_REDUCE',1)
          CALL     MPI_REDUCE(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) 
          call timer_mpi('MPI_REDUCE',2)
        END SUBROUTINE MPI_REDUCE_T
        
        SUBROUTINE MPI_ALLREDUCE_T(                                     &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_ALLREDUCE
          call timer_mpi('MPI_ALLREDUCE',1)
          CALL     MPI_ALLREDUCE(                                       &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          call timer_mpi('MPI_ALLREDUCE',2)
        END SUBROUTINE MPI_ALLREDUCE_T
        
        SUBROUTINE MPI_REDUCE_SCATTER_T(                                &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: RECVCOUNTS(*)
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_REDUCE_SCATTER
          CALL     MPI_REDUCE_SCATTER(                                  &
     &      SENDBUF, RECVBUF, RECVCOUNTS, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_REDUCE_SCATTER_T
        
        SUBROUTINE MPI_SCAN_T(                                          &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
          character(*), INTENT(IN)  :: SENDBUF(*)
          character(*), INTENT(OUT) :: RECVBUF
          INTEGER, INTENT(IN)  :: COUNT
          INTEGER, INTENT(IN)  :: DATATYPE
          INTEGER, INTENT(IN)  :: OP
          INTEGER, INTENT(IN)  :: COMM
          INTEGER, INTENT(OUT) :: IERROR 
          EXTERNAL MPI_SCAN
          CALL     MPI_SCAN(                                            &
     &      SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) 
        END SUBROUTINE MPI_SCAN_T
       
      END MODULE MPI__character_VS
