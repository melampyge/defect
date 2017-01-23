
""" Recursion to check friend points around the main point"""

##############################################################################
 
import numpy as np
import find_defects

##############################################################################

def recursion(defect_cluster_list, friend_list, dmax, cc, \
		rn, nn, x, y, phi_nematic, \
		nseg, cid, sim, cell_list, possible_defect_pts, \
		pt_colors, fig_cnt, total_rec_num, rec_num):
    """ check friend of main point as well as friends of friends and so on"""
    
    if total_rec_num == rec_num:
        return
        
    else:
        rec_num = rec_num + 1
        
        ### probe friend list
        
        for xd_friend, yd_friend in friend_list:
            
            dmax_n = calculate_defect_recursion(xd_friend, yd_friend, \
                                                x, y, phi_nematic, nseg, cid, sim, \
                                                cell_list, possible_defect_pts, pt_colors, \
                                                rn, nn, fig_cnt)
            
            ### for +1/2 defects
            
            if cc=='g' and np.abs(dmax-0.5) > np.abs(dmax_n-0.5):
                
                pt_colors.append(cc)
                defect_cluster_list.append([xd_friend, yd_friend, dmax_n])
                #print "Improved dmax in recursion level", rec_num, "is:", dmax_n
                friend_list_rec = find_friends(dmax_n, xd_friend, yd_friend, \
                                                cell_list.rcut, rn, nn)
    	
            ### for -1/2 defects
            
            elif cc == 'r' and np.abs(dmax+0.5) > np.abs(dmax_n+0.5):	
                
                pt_colors.append(cc)
                defect_cluster_list.append([xd_friend, yd_friend, dmax_n])
                #print "Improved dmax in recursion level", rec_num, "is:", dmax_n
                friend_list_rec = find_friends(dmax_n, xd_friend, yd_friend, \
                                                cell_list.rcut, rn, nn)
    
            else:
                continue	

            recursion(defect_cluster_list, friend_list_rec, dmax_n, cc, \
                      rn, nn, x, y, phi_nematic, nseg, \
                      cid, sim, cell_list, possible_defect_pts, pt_colors, \
                      fig_cnt, total_rec_num, rec_num)

    return
    
##############################################################################            

def find_friends(dmax, xd, yd, rcut, inner_radius, num):   
    """ find friends around the point within a band between inner_radius and rcut"""
    
    t = np.random.uniform(0.0, 2.0*np.pi, num)
    r = np.sqrt(np.random.uniform(inner_radius, rcut, num))
    
    return np.array(zip(xd + r * np.cos(t), yd + r * np.sin(t)))
    
##############################################################################    

def calculate_defect_recursion(xd, yd, x, y, phi_nematic, nseg, cid, sim, cell_list, \
			  possible_defect_pts, pt_colors, rn, nn, fig_cnt):
    """ calculate the defect strength of the friend point"""

    ### allocate array to divide the full circle into orthants
    
    nseg = 10

    ### calculate and average the order parameter matrix per orthant
    
    qxx, qxy, qyy, xcm, ycm = find_defects.calculate_order_param_matrix(xd, yd, nseg, \
                                                                        x, y, phi_nematic, sim, cell_list)
    
    ### calculate the nematic directors per orthant
    
    directors, corrected_directors = find_defects.calculate_nematic_directors(qxx, qxy, qyy, nseg)

    ### determine the defect strength
    
    dmax = find_defects.calculate_defect_strength(corrected_directors)

    return dmax

##############################################################################
    
def hit_gold(defect_strength_cut_max,dmax,cc):
    """ legacy function"""
    
    ddmax = dmax
    DEFECT_BOOL = True
    if (dmax>(-defect_strength_cut_max-0.5) and dmax<(defect_strength_cut_max-0.5)):
        ddmax = dmax
        cc = 'r'
        DEFECT_BOOL = False
    elif (dmax>(0.5-defect_strength_cut_max) and dmax<(defect_strength_cut_max+0.5)):
        ddmax = dmax
        cc = 'g'
        DEFECT_BOOL = False
        
    return ddmax, cc, DEFECT_BOOL

##############################################################################
    
def hit_gold2(defect_strength_cut_max,dmax_n, dmax, cc, xd, yd, xd_friend,yd_friend):
    """ legacy function 2"""
    
    ddmax = dmax
    BREAK_ALL = False;
    if (dmax_n>-defect_strength_cut_max-0.5 and dmax_n<defect_strength_cut_max-0.5):
        BREAK_ALL = True;
        cc = 'r'
        xd, yd, ddmax = xd_friend, yd_friend, dmax_n
    
    elif (dmax_n>0.5-defect_strength_cut_max and dmax_n<defect_strength_cut_max+0.5):
        BREAK_ALL = True;
        cc = 'g'
        xd, yd, ddmax = xd_friend, yd_friend, dmax_n
    
    return xd, yd, ddmax, cc, BREAK_ALL
            
    
##############################################################################   
