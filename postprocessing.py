import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
colors=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]

######################################
def find_gamma(path):
    f=open(path+'/output/harmonic_properties.dat','r')
    lines=f.readlines()
    f.close()
    for i in range(len(lines)):
        if 'nan' in lines[i]:
            return i-1

#########################################


def load_data(file,flag):
    """
    Takes file reads data from nth line onwards.
    nth line indicated by flag
    Example: for harmonic_properties.dat, read from 2nd line onwards, so flag=1
    """
    data = []
    with open(file, "r") as f:
        lines = f.readlines()
        if flag==-1:
            return lines[0]
        for line in lines[flag:]:
            val = [float(_) for _ in line.split()]
            data.append(val)
    data = np.array(data)       
    data[np.isnan(data)] = 0.0
    data[np.isinf(data)] = 1e30
    
    return data

######################################

def files(func):  
    """
    Takes path, reads files in output folder and returns their data
    """
    path = func + '/output/'
        
    if os.path.exists(path+'harmonic_properties.dat'):
        harmonic = load_data(path+'harmonic_properties.dat',1)
    else:
        harmonic=-1
    if os.path.exists(path+'directional_harmonic_properties.dat'):
        directional = load_data(path+'directional_harmonic_properties.dat',1)
    else:
        directional=-1
    if os.path.exists(path+'scattering_phase_space_3.dat'):
        space3 = load_data(path+'scattering_phase_space_3.dat',0)
    else:
        space3=-1
    if os.path.exists(path+'scattering_phase_space_4.dat'):
        space4 = load_data(path+'scattering_phase_space_4.dat',0)
    else:
        space4=-1
    if os.path.exists(path+'phonon_iterative.dat'):
        phonon_iter = load_data(path+'phonon_iterative.dat',1)
    else:
        phonon_iter=-1
    if os.path.exists(path+'predictedPhonon_RTA.dat'):
        rta_info=load_data(path+'predictedPhonon_RTA.dat',-1)
        phonon_rta = load_data(path+'predictedPhonon_RTA.dat',1)
    else:
        if os.path.exists(path+'phonon_RTA.dat'):
            rta_info=load_data(path+'phonon_RTA.dat',-1)
            phonon_rta = load_data(path+'phonon_RTA.dat',1)
        else:
            phonon_rta=-1
            rta_info=-1
    if os.path.exists(path+'pdos.dat'):
        phonon_dos = load_data(path+'pdos.dat',1)
    else:
        phonon_dos=-1
        dos_info=-1        
    if os.path.exists(path+'phonon_coherent.dat'):
        coherent = load_data(path+'phonon_coherent.dat',1)
    else:
        coherent=-1
    return harmonic, directional, phonon_rta, phonon_iter, space3, space4, phonon_dos,rta_info,coherent

################################################################################

def properties(func):
    """
    Takes path and returns phonon harmonic and anharmonic properties.
    """
    prop = {}
    
    harmonic, directional, phonon_rta, phonon_iter, space3, space4, phonon_dos ,rta_info,coherent = files(func)
    
    if not isinstance(harmonic, int): # harmonic_properties.dat
        qx = harmonic[:,2] # Direct qpoint x coordinate
        qy = harmonic[:,3]
        qz = harmonic[:,4]
        q = np.sqrt(harmonic[:,2]**2+harmonic[:,3]**2+harmonic[:,4]**2)
        f = harmonic[:,8] # Phonon frequency
        c = harmonic[:,9] # Phonon heat capacity
        vx = harmonic[:,10] # Phonon group velocity-x
        vy = harmonic[:,11] # Phonon group velocity-y
        vz = harmonic[:,12]  # Phonon group velocity-z
        grx = harmonic[:,-3]  # Phonon Grüneisen parameter-x
        gry = harmonic[:,-2]  # Phonon Grüneisen parameter-y
        grz = harmonic[:,-1]  # Phonon Grüneisen parameter-x
        prop['qx']=qx ; prop['qy']=qy ; prop['qz']=qz ; prop['q']=q;
        prop['f']=f; prop['c']=c; prop['vx']=vx; prop['vy']=vy; prop['vz']=vz ;
        prop['grx']=grx;prop['gry']=gry;prop['grz']=grz 
    
    if not isinstance(directional, int): # directional_harmonic_properties.dat
        q_dir = directional[:,8] # Distance from Γ qpoint
        q_dir = q_dir/q_dir[-1]  
        f_dir = directional[:,9] # Phonon frequency
        grx_dir = directional[:,14] # Phonon Grüneisen parameter-x
        gry_dir = directional[:,15] # Phonon Grüneisen parameter-y
        grz_dir = directional[:,16] # Phonon Grüneisen parameter-z
        prop['q_dir']=q_dir; prop['f_dir']=f_dir; prop['grx_dir']=grx_dir;prop['gry_dir']=gry_dir;prop['grz_dir']=grz_dir
    
    if not isinstance(phonon_iter, int):
        tau_x = phonon_iter[:,-3] # Phonon lifetime-x
        tau_y = phonon_iter[:,-2] # Phonon lifetime-y
        tau_z = phonon_iter[:,-1] # Phonon lifetime-z
        prop['tau_x']=tau_x; prop['tau_y']=tau_y; prop['tau_z']=tau_z
    
    if not isinstance(phonon_rta, int):
        num_tau=len(rta_info.split('|'))
        tau=1/np.sum([1/phonon_rta[:,x] for x in range(2,num_tau)],axis=0)          
        prop['tau']=tau
        
    fqcell=atoms_in_input(func)[1]   
    
    if not isinstance(space3, int):
        space3 = space3[:,2]
        prop['space3']=space3/np.product(fqcell) # Phase Normalized space-3. SPS3=delta/[(3N)^2*FQCELL] 
                                                 #(3N)^2 factor already incorporated in ALD
        
    if not isinstance(phonon_dos, int):
        prop['pdos']=phonon_dos # Phonon DOS

    if not isinstance(coherent, int):
        prop['coherent']=coherent # Coherent Contribution
        
    if not isinstance(space4, int):
        space4 = space4[:,2]
        prop['space4']=space4/(np.product(fqcell))**2 # Phase Normalized space-4. SPS4=delta/[(3N)^3*FQCELL^2] 
        
    
    return prop
#############################################################################################

def plot_directional_dispersion(functions,lim=False,scatter=True,ylow=-1,yhigh=40,xlow=-0.5,xhigh=1.2,size= (6,6),save=False,savefilename="Dispersion.svg"):
    """
    Plots Directional Phonon Dispersions of multiple compounds.
    Input: functions,lim=False,scatter=True,ylow=-1,yhigh=40,xlow=-0.5,xhigh=1.2,size=(6,6),save=False,savefilename="Dispersion.svg"
    functions=['path1','path2'] 
    scatter=True (If use scatter plot)
    lim=True (If you want to use xlim and ylim for plot)
    Provide limiting values:  ylow=-1,yhigh=40,xlow=-0.5,xhigh=1.2 
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Dispersion.svg" (Filename you want to save.)
    """

    plt.figure(figsize=size)
    plt.ylabel('Frequency ${\omega}$ (THz)',fontsize=10)
    plt.xlabel('Wave vector q ',fontsize=10)
    plt.title('Phonon Dispersion',fontsize=10)
    for i,func in enumerate(functions):
        prop = properties(func)
        if scatter:
            plt.scatter(prop['q_dir'],prop['f_dir'],s=2,label=func,color=colors[i])
        else:
            nbranch=3*atoms_in_input(func)[0]
            plt.plot(prop['q_dir'][::nbranch],prop['f_dir'][::nbranch],label = func,color=colors[i])
            for j in range(1,nbranch):
                plt.plot(prop['q_dir'][j::nbranch],prop['f_dir'][j::nbranch],color=colors[i])
    plt.tick_params(direction='in',right=True,bottom=True)
    if lim:
        plt.xlim(xlow,xhigh)
        plt.ylim(ylow,yhigh)
    plt.legend()
    if save:
        plt.savefig(savefilename)
    plt.show()  
    
#################################################################

def atoms_in_input(func):
    """
    Returns useful parameters defined in input.dat
    natoms,qgrid,Participating atoms
    """
    f=open(func+'/input.dat','r')
    lines=f.readlines()
    f.close()
    for i in range(len(lines)):
        if 'natom' in lines[i].split():
            idx=i
            break
    natoms=int(lines[idx].split()[-1])
    for i in range(len(lines)):
        if 'fqcell' in lines[i].split():
            idx2=i
            break 
    qgrid=[int(x) for x in lines[idx2].split()[-3:]]
    symbols=[]
    for i in range(natoms):
        symbols.append(lines[idx+1+i].split()[0])


    return np.array([natoms,qgrid,symbols])



#################################################################

def plot_pdos(path,size=(6,6),flag='Both',save=False,savefilename="DoS.svg"):
    """
    Plots Phonon Density Of States of given compound.
    Input:path,size=(6,6),flag='Both',save=False,savefilename="Dispersion.svg"
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    flag='Both','Total','Individual'
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="DoS.svg" (Filename you want to save.)
    """
    prop = properties(path)
    freq = prop['pdos'][:,0]
    dos_total = prop['pdos'][:,-1]
    # Read atoms from input.dat
    symbols=atoms_in_input(path)[-1]
    dos=[]
    sym=np.unique(symbols)
    for i in sym:
        pdos_sum=np.zeros(len(prop['pdos'][:,0]))
        for j,k in enumerate(symbols):
            if i==k:
                pdos_sum+=prop['pdos'][:,(j+1)]
        dos.append(pdos_sum)
    dos=np.array(dos) 
    dos=np.where(dos>1,1.0,dos)
    dos_total = np.expand_dims(dos_total, 0)
    freq = np.expand_dims(freq, 0)
    data=np.concatenate((freq,dos,dos_total))
    f = np.linspace(np.min(freq), np.max(freq), 1000)
    bandwidth = 0.05
    sum_=[];kde=[]
    for s in range(len(sym)):
        KDE=scipy.stats.gaussian_kde(data[0], weights=data[1+s])
        KDE.covariance_factor = lambda : bandwidth
        KDE._compute_covariance()
        kde.append(KDE(f))
        sum_.append(sum(data[1+s]))
    plt.figure(figsize=size)
    sum_=np.array(sum_)
    kde=np.array(kde)
    if flag=='Total' or flag=='total' or flag=='TOTAL':
        temp=np.zeros(len(f))
        for j in range(len(sym)):
            temp+=kde[j]*sum_[j]
        plt.plot(temp,f,color=colors[0])
        plt.legend(['Total'])
    elif flag=='Individual' or flag=='individual' or flag=='INDIVIDUAL':         
        for j in range(len(sym)):
            plt.plot(kde[j]*sum_[j],f,color=colors[j])
        plt.legend(sym)
    else:
        temp=np.zeros(len(f))
        for j in range(len(sym)):
            plt.plot(kde[j]*sum_[j],f,label=sym[j],color=colors[j])
            temp+=kde[j]*sum_[j]
        plt.plot(temp,f,label='Total',color=colors[j+1])
        plt.legend()  
    plt.xticks([])
    plt.tick_params(direction='in',right=False,left=True)
    plt.ylabel('Frequency ${\omega}$ (THz)',fontsize=10)
    if save:
        plt.savefig(savefilename)
    plt.show() 

#################################################################

def plot_heat_capacity(functions,size=(6,6),factors=1.0,save=False,savefilename="Heat_Capacity.svg"):
    """
    Plots Phonon Heat Capacity of multiple compounds.
    Input: functions,size=(6,6),save=False,savefilename="Heat_Capacity.svg"
    functions=['path1','path2']
    factors: [factor1,factor2] Multiplying factor for ALD reported thermal-K(for monolayer): default=1.0
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Heat_Capacity.svg" (Filename you want to save.)
    """
    plt.figure(figsize=size)
    plt.xlabel('Frequency ${\omega}$ (THz)',fontsize=10)
    plt.ylabel('Heat Capacity ${(J/m^3.K)}$',fontsize=10)
    plt.title('Phonon Heat Capacity',fontsize=10)
    for j,func in enumerate(functions):
        prop = properties(func)
        if factors==1:
            plt.scatter(prop['f'],prop['c'],s=2,label=func,color=colors[j])
        elif type(factors) == list and len(factors)==len(functions):
            plt.scatter(prop['f'],factors[j]*prop['c'],s=2,label=func,color=colors[j])
        else:
            return "Check factors input."
    plt.legend()
    plt.tick_params(direction='in',right=True,bottom=True)
    if save:
        plt.savefig(savefilename) 
    plt.show()
    
###################################################################################

def plot_grp_velocity(functions,axes=[0],size=(6,6),save=False,savefilename="Group_Velocity.svg"):
    """
    Plots absolute Phonon Group Velocity of multiple compounds in given axis.
    Input: functions,axes=[0],size=(6,6),save=False,savefilename="Group_Velocity.svg"
    functions=['path1','path2']
    axes=[0,1,2] for x y and z direction respectively
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Group_Velocity.svg" (Filename you want to save.)
    """
    axis_name=['x','y','z']
    plt.figure(figsize=size)
    plt.xlabel('Frequency ${\omega}$ (THz)',fontsize=10)
    plt.ylabel('Phonon-Group Velocity (m/s)',fontsize=10)
    plt.title('Phonon Group velocity',fontsize=10)
    for axis in axes:
        for j,func in enumerate(functions):
            prop = properties(func)
            plt.scatter(prop['f'],abs(prop['v'+axis_name[axis]]),s=2,label=func+'_'+axis_name[axis],color=colors[j])
    plt.legend()
    plt.tick_params(direction='in',bottom=True,left=True)
    if save:
        plt.savefig(savefilename)
    plt.show() 

    
###################################################################################    
    

def accumulated_coherent_thermal_K(path):
    """
    Returns accumulated coherent contribution to thermal conductivity vs frequency
    input:path
    """    
    prop = properties(path)['coherent']
    f=open(path+'/output/phonon_coherent.dat')
    lines=f.readlines()
    f.close()
    xx=prop[:,2];yy=prop[:,6];zz=prop[:,-1]
    f=properties(path)['f']
    f, xx,yy,zz = zip(*sorted(zip(f, xx,yy,zz)))
    kx=[xx[0]];ky=[yy[0]];kz=[zz[0]]
    for i in range(1,len(f)):
        kx.append(xx[i]+kx[i-1])
        ky.append(yy[i]+ky[i-1])
        kz.append(zz[i]+kz[i-1])
    return f,kx,ky,kz


#######################################################################################

def plot_scattering_lifetime(functions,axes=[0],ITER=False,lim=False,xlow=-0.5,xhigh=35,ylow=1e-12,yhigh=1e-8,ylog=False,size=(6,6),save=False,savefilename="Scattering_lifetime.svg"):
    """
    Plots absolute Phonon Group Velocity of multiple compounds in given axis.
    Input: functions,axes=[0],ITER=False,lim=False,xlow=-0.5,xhigh=35,ylow=1e-12,yhigh=1e-8,size=(6,6),save=False,savefilename="Scattering_lifetime.svg"
    functions=['path1','path2']
    ITER=False Iteratively calculated phonon-lifetimes
    axes=[0,1,2] for x y and z direction respectively
    ylog=False If use log scale for lifetimes.
    lim=True (If you want to use xlim and ylim for plot)
    Provide limiting values: ,xlow=-0.5,xhigh=35,ylow=1e-12,yhigh=1e-8,
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Scattering_lifetime.svg" (Filename you want to save.)
    """
    axis_name=['x','y','z']
    plt.figure(figsize=size)
    plt.xlabel('Frequency ${\omega}$ (THz)',fontsize=10)
    if ITER:
        plt.ylabel('Iterative Scattering lifetime (s)',fontsize=10)
        for axis in axes:
            for j,func in enumerate(functions):
                prop = properties(func)
                plt.scatter(prop['f'],prop['tau_'+axis_name[axis]],s=2,label=func+'_'+axis_name[axis],color=colors[j])
    else: 
        plt.ylabel('RTA Scattering lifetime (s)',fontsize=10)
        for j,func in enumerate(functions):
            prop = properties(func)
            plt.scatter(prop['f'],prop['tau'],s=2,label=func,color=colors[j])
    if ylog:
        plt.yscale('log')
    if lim:
        plt.ylim(ylow,yhigh)
        plt.xlim(xlow,xhigh)
    plt.legend()
    if save:
        plt.savefig(savefilename)
    plt.show() 
    
####################################################################
def eigenvectors(path):
    """
    Returns eigenvectors of phonons 
    Input: path
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    """
    f=open(path+'/output/eigenVector.dat','r')
    lines=f.readlines()
    f.close()
    natoms=atoms_in_input(path)[0]
    eigenvector=[]
    for j in range(1,len(lines)):
        val=[]
        for i in range(3*natoms):
            temp=lines[j].split('(')[1+i].split(')')[0].split(',')
            val.append(np.array(float(temp[0])+float(temp[1])*np.array(0+1j)))
        eigenvector.append(val)
    return np.array(eigenvector)
##############################################################################################
def accumulated_thermal_K(path,ITER=False,axis=0,normalized=False,mfp=False,w_cap=0):
    """
    Returns Phonon frequency/MFP of given compound and ALD-reported accumulated thermal conductivity.
    Input: path,ITER=False,axis=0,normalized=False,mfp=False,w_cap=0
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    ITER=False Iteratively calculated phonon-thermal conductivity.
    axis=0 for x direction.(Note: Only one value is accepted.)
    normalized=False (True: Provides normalized phonon thermal conductivity.)
    mfp=False (True: Provides phonon Mean Free Path in (nm).)
    w_cap=0 No Cap, 
  ( For non-zero w_cap: If MFP=True (specify value in nm) Ex. w_cap=50 All phonons MPF > 50nm contribute 0 to phonon thermal conductivity.
    If MFP=False: Ex. w_cap=5 : All phonons with frequencies > 5 THz contribute 0 to phonon thermal conductivity.)
    """
    axis_name=['x','y','z']
    prop = properties(path)
    if mfp:
        v=(prop['vx']**2 + prop['vy']**2 + prop['vz']**2)**0.5
        if ITER:
            w=1e9*v*prop['tau_'+axis_name[axis]]
        else:
            w=1e9*v*prop['tau']
    else:
        w=prop['f']
    if ITER:
        K=[prop['vx']**2*prop['c']*prop['tau_x'],prop['vy']**2*prop['c']*prop['tau_y'],prop['vz']**2*prop['c']*prop['tau_z']]
    else:
        K=[prop['vx']**2*prop['c']*prop['tau'],prop['vy']**2*prop['c']*prop['tau'],prop['vz']**2*prop['c']*prop['tau']]
    thermalk=np.array(K[axis])
    w,thermalk = zip(*sorted(zip(w,thermalk)))
    w=np.array(list(w))
    thermalk=np.array(list(thermalk))
    if w_cap !=0:
        thermalk=np.where(w<w_cap,thermalk,0)
    k_acc=[thermalk[0]]
    for i in range(1,len(thermalk)):
        k_acc.append(thermalk[i]+k_acc[i-1])
    if normalized:
        k_acc=k_acc/k_acc[-1]
    return w,k_acc
##########################################################################

def plot_accumulated_thermal_K(functions,ITER=False,axes=[0],normalized=False,mfp=False,w_cap=0,factors=1,size=(6,6),save=False,savefilename="Accumulated_thermal_K.svg"):
    """
    Plots phonon frequency/MFP vs accumulated thermal conductivity of given compounds and given axes.
    Input: functions,ITER=False,axes=[0],normalized=False,mfp=False,w_cap=0,size=(6,6),save=False,savefilename="Accumulated_thermal_K.svg"
    functions=['path1','path2']
    ITER=False Iteratively calculated phonon-thermal conductivity.
    axis=0 for x direction.(Note: Only one value is accepted.)
    normalized=False (True: Provides normalized phonon thermal conductivity.)
    mfp=False (True: Provides phonon Mean Free Path in (nm).)
    w_cap=0 No Cap, 
  ( For non-zero w_cap: If MFP=True (specify value in nm) Ex. w_cap=50 All phonons MPF > 50nm contribute 0 to phonon thermal conductivity.
    If MFP=False: Ex. w_cap=5 : All phonons with frequencies > 5 THz contribute 0 to phonon thermal conductivity.)
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Accumulated_thermal_K.svg" (Filename you want to save.)
    """
    axis_name=['x','y','z']
    plt.figure(figsize=size)
    if ITER:
        plt.ylabel('Iterative Thermal Conductivity (W/mK)',fontsize=10)
    else:
        plt.ylabel('RTA Thermal Conductivity (W/mK)',fontsize=10)
    for axis in axes:
        for j,func in enumerate(functions):
            w,K=accumulated_thermal_K(func,ITER=ITER,axis=axis,normalized=normalized,mfp=mfp,w_cap=w_cap)
            w=np.array(w)
            K=np.array(K)
            if factors==1:
                K*=1
            elif type(factors) == list and len(factors)==len(functions):
                K=factors[j]*K
            else:
                return "Check factors input."
            plt.plot(w,K,label=func+'_'+axis_name[axis],color=colors[j])
    if mfp:
        plt.xlim(1e0,1e4)
        plt.xscale('log')
        plt.tick_params(direction='in',left=True,right=False,top=True,bottom=True,width=0.5,which='minor')
        plt.xlabel('Phonon MFP (nm)',fontsize=10)
    else:
        plt.xlabel('Frequency ${\omega}$ (THz)',fontsize=10)
    plt.tick_params(direction='in',left=True,right=True,top=True,bottom=True,width=0.5)    
    plt.legend()
    if save:
        plt.savefig(savefilename)
    plt.show() 
    
###################################################################################################################

def thermal_K(path,ITER=False):
    """
    Returns ALD-reported Thermal Conductivity [Kxx,Kyy,Kzz]
    Input: path,ITER=False
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    ITER=False Iteratively calculated phonon-thermal conductivity.
    """
    prop = properties(path)
    if ITER:
        K=[prop['vx']**2*prop['c']*prop['tau_x'],prop['vy']**2*prop['c']*prop['tau_y'],prop['vz']**2*prop['c']*prop['tau_z']]
    else:
        K=[prop['vx']**2*prop['c']*prop['tau'],prop['vy']**2*prop['c']*prop['tau'],prop['vz']**2*prop['c']*prop['tau']]        
    return np.array([np.sum(x) for x in np.array(K)])

########################################################################################################################

def plot_thermal_K(functions,axes=[0,1,2],ITER=False,size=(6,6),save=False,savefilename="Thermal-K-Bar.svg"):
    """
    Plots ALD-reported Thermal Conductivity [Kxx,Kyy,Kzz]
    Input: functions,axes=[0,1],ITER=False,size=(6,6),save=False,savefilename="Thermal-K-Bar.svg"
    functions=['path1','path2']
    ITER=False Iteratively calculated phonon-thermal conductivity.
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Thermal-K-Bar.svg" (Filename you want to save.)
    """
    plt.figure(figsize=size)
    if ITER:
        plt.suptitle('Iterative Thermal Conductivity (W/mK)',fontsize=10)
    else:
        plt.suptitle('RTA Thermal Conductivity (W/mK)',fontsize=10)
    KT=['Kxx','Kyy','Kzz']
    for j,plot in enumerate(axes):
        K=[thermal_K(x,ITER=ITER)[plot] for x in functions]
        plt.subplot(1, 3, plot+1)
        plt.bar(functions, height=K)
        plt.title(KT[plot])
    plt.xlabel("")
#    plt.legend(functions)
    if save:
        plt.savefig(savefilename)

########################################################################

def plot_SPS(functions,size=(6,6),flag=3,lim=False,ylow=-10,yhigh=200,xlow=-2,xhigh=40,save=False,savefilename="SPS-"):
    """
    Plots Scattering Phase Space (3/4) 
    Input: functions,flag=3,lim=False,ylow=-10,yhigh=200,xlow=-2,xhigh=40,save=False,savefilename="SPS-"
    functions=['path1','path2']
    flag=3 (3 for SPS-3 and 4 for SPS-4) 
    lim=True (If you want to use xlim and ylim for plot)
    Provide limiting values: ,ylow=-10,yhigh=200,xlow=-2,xhigh=40
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="SPS-" (Filename you want to save.)    
    """
    plt.figure(figsize=size)
    plt.xlabel('Frequency ${\omega}$ (THz)',fontsize=10)
    plt.ylabel('Scattering-Phase Space',fontsize=10)
    plt.title('Phonon Scattering-Phase Space-'+str(flag),fontsize=10)
    for j,func in enumerate(functions):
        prop = properties(func)
        plt.scatter(prop['f'],(prop['space'+str(flag)]),s=2,label=func,color=colors[j])
    if lim:
        plt.ylim(ylow,yhigh)
        plt.xlim(xlow,xhigh)
    plt.legend()
    if save:
        plt.savefig(savefilename+str(flag)+".svg")
    plt.show() 
    
########################################################################################

def plot_GR(functions,axis=0,lim=False,ylow=-30,yhigh=10,xlow=-2,xhigh=40,size=(6,6),save=False,savefilename="Gruniessen Parameter-x.svg"):
    """
    Plots Grüneisen parameters (Obtained from specified high symmetry points)
    Input: functions,axis=0,lim=False,ylow=-30,yhigh=10,xlow=-2,xhigh=40,save=False,savefilename="Gruniessen Parameter-"
    functions=['path1','path2']
    axis=0 for x direction (1/2 for y/z axis respectively)
    lim=True (If you want to use xlim and ylim for plot)
    Provide limiting values: ylow=-30,yhigh=10,xlow=-2,xhigh=40
    size=(6,6) plotsize: 6"x6"
    save=True (If you want to save the file.)
    savefilename="Gruniessen Parameter-" (Filename you want to save.)    
    """
    axis_name=['x','y','z']
    plt.figure(figsize=size)
    plt.xlabel('Frequency ${\omega}$ (THz)',fontsize=10)
    plt.ylabel('Gruniessen Parameter-'+axis_name[axis],fontsize=10)
    plt.title('Gruniessen Parameter',fontsize=10)
    for j,func in enumerate(functions):
        prop = properties(func)
        plt.scatter(prop['f_dir'],(prop['gr'+axis_name[axis]+'_dir']),s=2,label=func,color=colors[j])
    plt.legend()
    if lim:
        plt.ylim(ylow,yhigh)
        plt.xlim(xlow,xhigh)
    if save:
        plt.savefig(savefilename+axis_name[axis]+".svg")
    plt.show()
    
    
#######################################################################################

def average_SPS(path,flag=3):
    """
    Returns heat capacity weighted Grüneisen parameter
    Input: path,flag
    flag=3 (3 for SPS-3 and 4 for SPS-4) 
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    """
    prop = properties(path)
    SPS=prop['space'+str(flag)]
    c=prop['c']
    SPS_avg=(np.sum(c*SPS)/np.sum(c))
    return SPS_avg


    
#######################################################################################

def average_velocity(path):
    """
    Returns heat capacity weighted phonon velocity
    Input: path
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    """
    prop = properties(path)
    v=np.sqrt((prop['vx']**2+prop['vy']**2+prop['vz']**2))
    c=prop['c']
    v_avg=(np.sum(c*v)/np.sum(c))
    return np.round(v_avg,2)
    
    
#######################################################################################

def average_GR(path):
    """
    Returns heat capacity weighted Grüneisen parameter
    Input: path
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    """
    prop = properties(path)
    gr=(abs(prop['grx'])+abs(prop['gry'])+abs(prop['grz']))/3
    c=prop['c']
    gr_avg=(np.sum(c*gr)/np.sum(c))
    return np.round(gr_avg,2)
    
    
#######################################################################################


def total_heat_capacity(path):
    """
    Prints Total heat capacity in J/Kg-K and J/mol-K
    Input: path
    path: path of input.dat and output directory of the compound. eg. '/home/MatSim/amey2618/2D/'
    """
    f=open(path+'/output/cell_conventional.dat','r') # Get conventional cell lattice parameters
    lines=f.readlines()
    f.close()
    C=np.zeros((3,3))
    for i in range(3):
        C[i]=[float(x) for x in lines[1+i].split()]
    V=np.dot(C[0],np.cross(C[1],C[2]))*1e-30 # Calculate volume of this cell
    mass=0.0
    for i in lines[4:]:
        mass+=float(i.split()[-4])
    f=open(path+'/input.dat','r')
    lines=f.readlines()
    f.close()
    for i in lines:
        if 'nmass' in i.split():
            mul=float(i.split()[-1])
            break
    Mass=mul*mass # Get total mass
    g=open(path+'/output/harmonic_properties.dat','r')
    lines=g.readlines()
    g.close()
    C=[]
    for i in lines[1:]:
        C.append(float(i.split()[9]))
    C=np.array(C)
    C[np.isnan(C)] = 0.0
    symbols=atoms_in_input(path)[-1]
    count1=100
    for i in np.unique(symbols):
        count=0
        for j,k in enumerate(symbols):
            if i==k:
                count+=1
        count1=min(count,count1)
    num_mol=max(1,count1) # Number of molecules of given compound
#    print("Number of molecules = ", num_mol) 
#    print('Total heat Capacity of '+path+' = ',np.round(np.sum(C)*V/Mass,2),' J/Kg-K')
#    print('Total heat Capacity of '+path+' = ',np.round(np.sum(C)/num_mol*V*6.0221408e23,2),' J/mol-K')
    return np.round(np.sum(C)/num_mol*V*6.0221408e23,2) # Heat capacity in J/mol-K

########################################################################################

def Summary(path):
    print('ALD-reported Thermal Conductivity- for '+path+" = ",np.round(thermal_K(path)[0],2),"W/m-K")
    print('Heat Capacity weighted-Gruniessen Parameter- for '+path+' = ',average_GR(path))
    print('Heat Capacity weighted-Scattering Phase Space-3 for '+path+' = ',average_SPS(path))
    print('Heat Capacity weighted-Velocity-'+' for '+path+' = ',average_velocity(path))
    print('Total heat Capacity of '+path+' = ',total_heat_capacity(path),' J/mol-K')


########################################################################################
