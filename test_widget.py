
import numpy as np
import os
import dill

import matplotlib.pyplot as plt
from sfiabp.display.sweep import sfidisp_sweep

#########################
## list_Sfile or Sfile
#########################

# PathOutFile = 'results/S_Trigo_Sim_1r4_npar_82_u_6_k_2000_5000f_Ito_pinv_Gwn_0_Order_1_1000f.pkl'
# # PathOutFile= 'results/List_S_Trigo_Sim_1r4_npar_82_u_6_k_2000_5000f_Ito_pinv_Gwn_0_PolyExp_Order_1.pkl'
# # PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/List_S_Trigo_Sim_SSP_5000f_Strato_pinv_Gwn_0_PolyExp_Order_2.pkl'

# # with open( PathOutFile, 'rb' ) as inp:    
# #     Sabp = dill.load(inp)

# # # plottrigo.display_v1(PathOutFile,funr = lambda r : 2000/r**4)
# # # vj = np.linspace(-np.pi,np.pi, num=32+1,endpoint=True)
# # # vj = np.linspace(0,2*np.pi, num=32+1,endpoint=True)
# # # plottrigo.sfidisplay_abp(PathOutFile,vj=vj,d=3.17,funr = lambda r : 2000/r**4)
# tishift = 0
# tjshift = -np.pi
# plottrigo.sfidisplay_abp(PathOutFile,d=3.17,tishift=tishift,tjshift=tjshift)

#########################
## directory
#########################

# PathOutFile= 'results/test'
# PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/List_S_Trigo_Sim_SSP_5000f_Strato_pinv_Gwn_0_PolyExp_Order_2.pkl'
# PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/exp_poncet'
# PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/bench_ssp'
# PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/test_tiko'
# list_PathOutFile = [ '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_order_nfra/bench_1r4_npar_484',
#                      '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_order_nfra/bench_ssp',
#                      '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_order_nfra/exp_poncet',
#                      '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_order_nfra/exp_nishi']
                     
# list_PathOutFile = [ '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/exp_poncet',
#                      '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/bench_ssp',
#                      '/media/sf_PostDoc_Gulliver/SFI/depot_v19/collect_data/list_strigo_nfra/bench_1r4' ]
# for path in list_PathOutFile:
#     plottrigo.plot_tutorial(PathOutFile)

tjshift = -np.pi
# PathOutFilePoncet = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/exp_poncet/sfi_list_order_nfra'
# PathOutFilePoncet = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/exp_nishi_order/sfi'
# PathOutFilePoncet = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/exp_poncet/sfi/polyexp_strato_pinv/S_Trigo_run_5khz_10vpp_Strato_pinv_lcell_24_Order_3_4002f.pkl'
# PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/exp_nishi_order/sfi'
PathOutFile = '/media/sf_PostDoc_Gulliver/SFI/depot_v19/exp_poncet/sfi_pinv'

sfidisp_sweep(PathOutFile,d=3.17,tjshift=tjshift,dijinc=0.1)
# FigManager = plt.get_current_fig_manager()
# FigManager.full_screen_toggle()
plt.show(block=False)
print('ok')