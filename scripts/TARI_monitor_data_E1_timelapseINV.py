# -*- coding: utf-8 -*-
# %%
from os import listdir
from os.path import isdir, join
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = "Microsoft Sans Serif"


output_path = r'D:\R2MSDATA\TARI_E1_test\output'
output_folders = sorted([f for f in listdir(output_path) if isdir(join(output_path,f))])

def check_files_in_directory(directory_path,Line_name = 'E1'):
    # 存儲解析出來的日期
    dates = []

    # 遍歷資料夾中的所有檔案
    for filename in listdir(directory_path):
        # 檢查檔案名稱是否符合特定格式
        if filename.endswith('_m_'+Line_name):
            date_str = filename[:8]  # 提取日期部分
            try:
                # 轉換日期格式從 'YYMMDDHH' 到 datetime 對象
                date = datetime.strptime(date_str, '%y%m%d%H')
                dates.append(date)
            except ValueError:
                # 如果日期格式不正確，忽略此檔案
                continue
    dates = sorted(dates)
    return dates

dates = check_files_in_directory(output_path,Line_name = 'E1')
# pick up the time of the index from 2024/2/29 21:00 to 2024/3/9 8:00
date_lim = [datetime(2024,10,31,0,0),datetime(2024,11,12,0,0)]
picked_date = [date for date in dates if date >= date_lim[0] and date <= date_lim[1]]
picked_date_index = [dates.index(date) for date in picked_date]
picked_output_folders = [output_folders[i] for i in picked_date_index]
# %%
def load_1inversion_data(save_ph):
    output_ph = join(save_ph,'ERTManager')
    # Load data file
    data_path = join(output_ph,'inverison_data.ohm')
    data = ert.load(data_path)
    investg_depth = (max(pg.x(data))-min(pg.x(data)))*0.25
    # Load model response
    resp_path = join(output_ph,'model_response.txt')
    response = np.loadtxt(resp_path)

    mgr_dict = {'data': data, 
                'response': response}

    return mgr_dict

all_mgrs = []
for i,output_folder_name in enumerate(picked_output_folders):
    print(output_folder_name)
    all_mgrs.append(load_1inversion_data(join(output_path,output_folder_name)))

# %%
filtered_mgrs = all_mgrs
DATA = []
for i,output_folder_name in enumerate(picked_output_folders):
    remain_per = 0.9
    t1 = np.argsort(all_mgrs[i]['data']['misfit'])[int(np.round(remain_per*len(all_mgrs[i]['data']['rhoa']),0)):]
    remove_index = np.full((len(all_mgrs[i]['data']['rhoa'])), False)
    for j in range(len(t1)):
        remove_index[t1[j]] = True
    print(r'remove {:d}% worst misfit data, rest data {:d}'.format(int(100*(1-remain_per)),len(all_mgrs[i]['data']['rhoa'])-len(t1)))
    filtered_mgrs[i]['data'].remove(remove_index)
    DATA.append(filtered_mgrs[i]['data'])
# %%
# Four electrode numbers for each data set
sets_of_quadruples = []

for data in DATA:
    quadruples = set(zip(data['a'], data['b'], data['m'], data['n']))
    sets_of_quadruples.append(quadruples)

# find the common quadruples
common_quadruples = set.intersection(*sets_of_quadruples)

# delete the data with quadruples not in common_quadruples
remove_indices_list = []
i = 0
for data in DATA:
    quadruples = list(zip(data['a'], data['b'], data['m'], data['n']))
    remove_indices = [quadruple not in common_quadruples for quadruple in quadruples]
    remove_indices_list.append(remove_indices)
    filtered_mgrs[i]['data'].remove(remove_indices)
    print(filtered_mgrs[i]['data'])
    i += 1
# ert.show(filtered_mgrs[i-1]['data'],filtered_mgrs[i-1]['data']['rhoa'])
# %%
def plot_inverted_profile(mgr, data, urf_file_name, lam, rrms, chi2, **kw):
    ax, cb = pg.show(mgr.paraDomain,mgr.model,
                    coverage=1, **kw)
    ax.plot(np.array(pg.x(data)), np.array(pg.z(data)), 'kd',markersize = 3)
    title_str = 'Inverted Resistivity Profile at {}number of data={:.0f}, rrms={:.2f}%, $\chi^2$={:.3f}, $\lambda$={:.0f}'.format(
        datetime.strptime(urf_file_name[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00")+'\n',len(data['rhoa']),rrms,chi2,lam)
    ax.set_title(title_str)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    fig = ax.figure

    cb_ytick_label = np.round(cb.ax.get_yticks(),decimals = 0)
    cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick_label])
    fig.savefig(join(output_ph,urf_file_name[0:-4],'INV_lamda{}.png'.format(lam)),dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_inverted_contour(mgr, data, urf_file_name, lam, rrms, chi2, fig, **kw):
    mesh_x = np.linspace(left, right, 250)
    mesh_y = np.linspace(-depth, 0, 150)
    grid = pg.createGrid(x=mesh_x, y= mesh_y)
    X,Y = np.meshgrid(mesh_x,mesh_y)
    rho_grid = np.reshape(pg.interpolate(mgr.paraDomain, np.log10(mgr.model), grid.positions()),(len(mesh_y),len(mesh_x)))

    fig, ax = plt.subplots(figsize=(fig.get_figwidth(), fig.get_figheight()))
    ax.contourf(X,Y, rho_grid, cmap=kw['cMap'], levels=50,
                vmin=np.log10(kw['cMin']),vmax=np.log10(kw['cMax']))
    ax.set_aspect('equal')
    ax.set_xlim(left,right)
    ax.set_ylim(-depth,0)
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=.15)
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    m.set_array(rho_grid)
    m.set_clim(np.log10(kw['cMin']),np.log10(kw['cMax']))
    cb = plt.colorbar(m, boundaries=np.linspace(np.log10(kw['cMin']),np.log10(kw['cMax']), 50),cax=cbaxes)
    cb.ax.set_yticks(np.linspace(np.log10(kw['cMin']),np.log10(kw['cMax']),5))
    cb.ax.set_yticklabels(['{:.0f}'.format(10**x) for x in cb.ax.get_yticks()])
    cb.ax.set_ylabel(kw['label'])
    title_str = 'Inverted Resistivity Profile at {}number of data={:.0f}, rrms={:.2f}%, $\chi^2$={:.3f}, $\lambda$={:.0f}'.format(
        datetime.strptime(urf_file_name[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00")+'\n',len(data['rhoa']),rrms,chi2,lam)
    ax.set_title(title_str)
    ax.set_xlabel(kw['xlabel'])
    ax.set_ylabel(kw['ylabel'])
    fig.savefig(join(output_ph, urf_file_name[0:-4], 'INV_contour.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_convergence(mgr, urf_file_name):
    rrmsHistory = mgr.inv.rrmsHistory
    chi2History = mgr.inv.chi2History
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(np.linspace(1,len(rrmsHistory),len(rrmsHistory)),rrmsHistory, linestyle='-', marker='o',c='black')
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('rRMS (%)')
    ax.set_title('Convergence Curve of Resistivity Inversion')
    ax2 = ax.twinx()
    ax2.plot(np.linspace(1,len(rrmsHistory),len(rrmsHistory)),chi2History, linestyle='-', marker='o',c='blue')
    ax2.set_ylabel('$\chi^2$',c='blue')
    ax.grid()
    fig.savefig(join(output_ph,urf_file_name[0:-4],'CONV.png'),dpi=300, bbox_inches='tight')
    plt.close(fig)

    return rrmsHistory, chi2History

def crossplot(mgr,urf_file_name):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(np.log10(mgr.data["rhoa"]),np.log10(mgr.inv.response),s=1)
    xticks = ax.get_xlim()
    yticks = ax.get_ylim()
    lim = max(max(yticks,xticks)) + 0.5
    ax.plot([0,lim],[0,lim],'k-',linewidth=1, alpha=0.2)
    ax.set_xlim([0,lim])
    ax.set_ylim([0,lim])
    ax.set_xlabel('Log10 of Measured Apparent resistivity')
    ax.set_ylabel('Log10 of Predicted Apparent resistivity')
    ax.set_title(r'Crossplot of Measured vs Predicted Resistivity $\rho_{apparent}$')
    fig.savefig(join(output_ph,urf_file_name[0:-4],'CROSP.png'),dpi=300, bbox_inches='tight')
    plt.close(fig)

def data_misfit(mgr, urf_file_name):
    mgr.data['misfit'] = np.abs((mgr.inv.response - mgr.data["rhoa"])/mgr.data["rhoa"])*100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(mgr.data['misfit'],np.linspace(0,100,21))
    ax.set_xticks(np.linspace(0,100,21))
    ax.set_xlabel('Relative Data Misfit (%)')
    ax.set_ylabel('Number of Data')
    ax.set_title('Data Misfit Histogram for Removal of Poorly-Fit Data')
    fig.savefig(join(output_ph, urf_file_name[0:-4], 'HIST.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def export_inversion_info(mgr, urf_file_name, lam, rrmsHistory, chi2History):
    information_ph = join(output_ph,urf_file_name[0:-4],'ERTManager','inv_info.txt')
    with open(information_ph, 'w') as write_obj:
        write_obj.write('## Final result ##\n')
        write_obj.write('rrms:{}\n'.format(mgr.inv.relrms()))
        write_obj.write('chi2:{}\n'.format(mgr.inv.chi2()))

        write_obj.write('## Inversion parameters ##\n')
        write_obj.write('use lam:{}\n'.format(lam))

        write_obj.write('## Iteration ##\n')
        write_obj.write('Iter.  rrms  chi2\n')
        for iter in range(len(rrmsHistory)):
            write_obj.write('{:.0f},{:.2f},{:.2f}\n'.format(iter,rrmsHistory[iter],chi2History[iter]))


import os
# second inversion output folder name: 
output_ph_second =output_path+'_second_inversion_'+date_lim[0].strftime("%m%d%H")+'_'+date_lim[1].strftime("%m%d%H")
# check if the folder exists
if not os.path.exists(output_ph_second):
    # if not, create the folder
    os.makedirs(output_ph_second)
    print(f'Folder "{output_ph_second}" created.')
else:
    print(f'Folder "{output_ph_second}" already exists.')


for i,output_folder_name in enumerate(picked_output_folders):
    pg.boxprint('Processing {:d} of {:d}: {:s}'.format(i+1, len(picked_output_folders), output_folder_name))

    if os.path.exists(join(output_ph_second,output_folder_name)):
        pg.boxprint(output_folder_name+'inversion is already processed. Skip it!')
        continue

    data = filtered_mgrs[i]['data']
    print(data)

    left = min(pg.x(data))
    right = max(pg.x(data))
    length = right - left
    depth = length/4
    print('Using first inversion mesh: '+picked_output_folders[i])
    previous_ph = join(output_path,picked_output_folders[i],'ERTManager','mesh.bms')
    mesh = pg.load(previous_ph)
    print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarkers() == 2) if x]))
    
    lam = 100
    mgr = ert.ERTManager(data)

    if i == 0:
        
        model = mgr.invert(data,mesh=mesh,
                            lam=lam  ,zWeight=1,
                            maxIter = 6,
                            verbose=True)
    else:
        pg.boxprint('Using previous inversion result as initial model: '+picked_output_folders[i-1])
        previous_ph = join(output_ph_second,picked_output_folders[i-1],'ERTManager','resistivity.vector')
        initial_model = pg.load(previous_ph)
        model = mgr.invert(data,mesh=mesh,startModel=initial_model,
                            lam=lam  ,zWeight=1,
                            maxIter = 6,
                            verbose=True)

    rrms = mgr.inv.relrms()
    chi2 = mgr.inv.chi2()
    pg.boxprint('rrms={:.2f}%, chi^2={:.3f}'.format(rrms, chi2))
    path, fig, ax = mgr.saveResult(join(output_ph_second,output_folder_name))
    plt.close(fig)

    # Plot the inverted profile
    kw = dict(label='Resistivity $\Omega m$',
                logScale=True,cMap='jet',cMin=32,cMax=3162,
                xlabel="x (m)", ylabel="z (m)",
                orientation = 'vertical')
    urf_file_name = output_folder_name + '.urf'
    output_ph = output_ph_second
    fig = plot_inverted_profile(mgr, data, urf_file_name, lam, rrms, chi2, **kw)

    # Plot inverted contour profile
    plot_inverted_contour(mgr, data, urf_file_name, lam, rrms, chi2, fig, **kw)

    # Convergence Curve of Resistivity Inversion
    rrmsHistory, chi2History = plot_convergence(mgr, urf_file_name)

    # Varify the fitted and measured data cross plot
    crossplot(mgr,urf_file_name)

    # Data Misfit Histogram for Removal of Poorly-Fit Data
    data_misfit(mgr, urf_file_name)
    
    # Export the information about the inversion
    export_inversion_info(mgr, urf_file_name, lam, rrmsHistory, chi2History)

    # Export data used in this inversion 
    mgr.data.save(join(output_ph_second,output_folder_name,'ERTManager','inverison_data.ohm'))
    # Export model response in this inversion 
    pg.utils.saveResult(join(output_ph_second,output_folder_name,'ERTManager','model_response.txt'),
                        data=mgr.inv.response, mode='w')  
# %%
