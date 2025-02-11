c
        program vertmat
c
c       VERTikaalmetingen naat Matfile
c
c       opzoeken meetgegevens uit files 1978 en 1979
c       op basis van x en y
c
        implicit none
c
	integer   idx,idx1,idx2,nregel
        real      xc,yc,xz,yz
        real      dx,dy,dafstand,dmarge
	character string*60,fnaam*64,cstr*1,als*1
	character ctemp*5,cchl*5,cdiep*2,fuit*64,cdat*6
        logical   lokat
c
c
	open (unit=20,file='vertmat.dat',status='old')
	read (20,*) als  
	read (20,*) als  
	read (20,*) als 
        read (20,'(t30,3f8.0)',end=999) xz,yz,dmarge
        read (20,'(t30,a)',end=999) fuit
        read (20,'(a)') als         
c
c
	open (unit=22,file=fuit,status='unknown')
c
        write (22,'(''# vertikaalmetingen Grevelingen'')')
        write (22,'(''# gevraagde lokatie :'')')
        write (22,'(''# X - : '',f8.0)') xz
        write (22,'(''# Y - : '',f8.0)') yz
        write (22,'(''# marge - : '',f8.0)') dmarge
        write (22,'(''$ -999 [0022]'')')
        write (22,'(''>Datum'')')
        write (22,'(''>Tijd '')')
        write (22,'(''>Diepte'')')
        write (22,'(''>Temperatuur '')')
        write (22,'(''>Chloride'')')
c
c
  100   read (20,'(a)',end=999) fnaam
	open (unit=21,file=fnaam,status='old')
c
c
  120   read (21,'(a)',end=888) string
        if (string(1:3) .eq. '+++') goto 888
c       write(*,*) string
	idx1 = 0
	idx2 = 0
	lokat =.false.
	if (string(1:5) .eq. 'Datum') then
         cstr='-'
	 idx1=index(string,cstr)
	 cdat(5:6)=string(idx1-2:idx1-1)
	 if (cdat(5:5) .eq. ' ') cdat(5:5)='0'
	 idx=index(string(idx1+1:),cstr)
	 idx1=idx1+idx
	 if (idx .eq. 2) then
           cdat(3:3)='0'
           cdat(4:4)=string(idx1-1:idx1-1)
	 endif
	 if (idx .eq. 3) then
           cdat(3:4)=string(idx1-2:idx1-1)
	 endif
	 cdat(1:2)=string(idx1+1:idx1+2)
        endif
c
	if (string(1:1) .eq. 'X') then
	 read(string(3:8),*) xc
	 read(string(13:18),*) yc
c test op gevraagde lokatie
         dx=xz-xc
	 dy=yz-yc
	 dafstand = sqrt(dx*dx + dy*dy)
	 if (dafstand .le. dmarge) lokat=.true.
c
        endif
	 if (.not.lokat) goto 120
c 
	read (21,'(a)') string
c       write(*,*) string
	if (string(:4) .eq. 'vlgn' ) then
	  idx1=index(string,'Temp')
	  idx2=index(string,'CL-g')
        endif
	if (idx1 .eq. 0 .and. idx2 .eq. 0) goto 120
c
c --- lezen meetdata
c
	nregel = 0 
  140   read (21,'(a)',end=888) string
c       write(*,*) string
        nregel=nregel+1
	if (string(1:3) .eq. '+++') goto 888
        if (string(1:4) .eq. 'Vert') goto 120
        cdiep =string(2:3)
        ctemp =string(idx1-1:idx1+3)
        cchl  =string(idx2-1:idx2+3)
	if (ctemp .eq. '$$$$$') ctemp = ' -999'
	if (cchl .eq. '$$$$$') cchl = ' -999'
c     
        if (nregel .eq. 1) then 
	write (22,'(a1,5a6)') '-',cdat,' 0 ',cdiep,ctemp,cchl
	else
	write (22,'(t2,5a6)') cdat,' 0 ',cdiep,ctemp,cchl
	endif
c
	goto 140
  888   close (unit=21)
	goto 100
  999   close (unit=20)
        close (unit=22)
	end
