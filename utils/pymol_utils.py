from pymol import cmd
import pymol
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm
import palettable
from tqdm import tqdm

# color_l = palettable.cartocolors.diverging.Temps_4.mpl_colors
color_l = palettable.cartocolors.diverging.Geyser_4.mpl_colors
# color_l = palettable.cartocolors.diverging.BlueDarkRed12_3.mpl_colors
color_0 = color_l[0]
color_1 = color_l[3]

def h_acceptor_distribution(results_fn_list):


    ligand_positions = []
    for index, ligand_file in enumerate(results_fn_list):
        print(index)
        # if index == 0:
        #     pymol.cmd.load(ligand_file, f'ligand')
        #     pymol.cmd.h_add("ligand")
        #     pymol.cmd.hide("everything", "ligand")
        #     pymol.cmd.show("sticks", "ligand")
        #     pymol.cmd.set("stick_radius", 0.15)
        #     pymol.cmd.util.cbaw("ligand")
        #     pymol.cmd.select("h_acceptor_index", "ligand and (elem o or (elem n and not (neighbor hydro)))")
        #     pymol.cmd.create("h_acceptor", "h_acceptor_index")
        # else:
        # ligand_name = os.path.splitext(os.path.basename(ligand_file))[0]
        # split_ligand_name = ligand_name.split('_')
        # dock_score = float(split_ligand_name[-1])
        # if dock_score > -7:
        #     clean_num  = clean_num + 1
        #     continue

        pymol.cmd.load(ligand_file, f'ligand{index}')
        pymol.cmd.h_add(f'ligand{index}')
        pymol.cmd.hide("everything", f'ligand{index}')
        pymol.cmd.show("sticks", f'ligand{index}')
        pymol.cmd.set("stick_radius", 0.15)
        pymol.cmd.util.cbaw(f'ligand{index}')

        # hydrophobic_atom_names = ["C*", "F*", "I*", "Br*", "Cl*", "S*"]
        # pymol.cmd.select("hydrophobic", "ligand and (name " + " +".join(hydrophobic_atom_names) + ")")

        pymol.cmd.select("h_acceptor_index", f"ligand{index} and (elem o or (elem n and not (neighbor hydro)))")

        model = pymol.cmd.get_model("h_acceptor_index")
        for atom in model.atom:
            ligand_positions.append(atom.coord)
        pymol.cmd.hide("everything", f'ligand{index}')

    pairs = np.array(ligand_positions)
    kde = gaussian_kde(pairs.T)
    pdf = kde.evaluate(pairs.T)
    norm = plt.Normalize(vmin=np.min(pdf), vmax=np.max(pdf))
    cmap = cm.get_cmap('YlOrRd')  # YlGnBu Oranges YlOrRd
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(pdf)

    def rgba_to_hex(rgba):
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        return f"0x{r:02x}{g:02x}{b:02x}"

    hex_colors = [rgba_to_hex(rgba) for rgba in colors]

    for index, position in enumerate(ligand_positions):
        object = f'o_{index}'
        cmd.pseudoatom(object, state=1, pos=position)
        pymol.cmd.hide("everything", object)
        cmd.show("spheres", object)
        pymol.cmd.set("sphere_scale", 0.1)
        pymol.cmd.set("sphere_transparency", 0)
        # pymol.cmd.set_color(f"custom_color_{index}", list(colors[index][:3]))
        pymol.cmd.set_color(f"custom_color_{index}", color_1)
        pymol.cmd.color(f"custom_color_{index}", object)



def hydrophobic_distribution(results_fn_list):


    ligand_positions = []
    for index, ligand_file in enumerate(results_fn_list):
        print(index)

        pymol.cmd.load(ligand_file, f'ligand{index}')
        pymol.cmd.h_add(f'ligand{index}')
        pymol.cmd.hide("everything", f'ligand{index}')
        pymol.cmd.show("sticks", f'ligand{index}')
        pymol.cmd.set("stick_radius", 0.15)
        pymol.cmd.util.cbaw(f'ligand{index}')

        pymol.cmd.select("methyl_groups", f"ligand{index} and name C and neighbor (name H and neighbor name C)")
        model = pymol.cmd.get_model("methyl_groups")
        for atom in model.atom:
            ligand_positions.append(atom.coord)

        pymol.cmd.select("methylene_groups", f"ligand{index} and name C and neighbor (name C and neighbor name H)")
        model = pymol.cmd.get_model("methylene_groups")
        for atom in model.atom:
            ligand_positions.append(atom.coord)

        pymol.cmd.select("benzene_ring", f"ligand{index} and (resn PHE+TYR+TRP+HIS) and (name CG+CD1+CD2+CE1+CE2+CZ)")
        model = pymol.cmd.get_model("benzene_ring")
        for atom in model.atom:
            ligand_positions.append(atom.coord)

        pymol.cmd.select("fluorine_atoms", f"ligand{index} and name F")
        model = pymol.cmd.get_model("fluorine_atoms")
        for atom in model.atom:
            ligand_positions.append(atom.coord)

        pymol.cmd.select("chlorine_atoms", f"ligand{index} and name Cl")
        model = pymol.cmd.get_model("chlorine_atoms")
        for atom in model.atom:
            ligand_positions.append(atom.coord)

        pymol.cmd.select("bromine_atoms", f"ligand{index} and name Br")
        model = pymol.cmd.get_model("bromine_atoms")
        for atom in model.atom:
            ligand_positions.append(atom.coord)

        pymol.cmd.hide("everything", f'ligand{index}')

    pairs = np.array(ligand_positions)
    kde = gaussian_kde(pairs.T)
    pdf = kde.evaluate(pairs.T)
    norm = plt.Normalize(vmin=np.min(pdf), vmax=np.max(pdf))
    cmap = cm.get_cmap('GnBu')
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors_1 = mappable.to_rgba(pdf)

    for index, position in enumerate(ligand_positions):
        object = f'o_{index+10000}'
        cmd.pseudoatom(object, state=1, pos=position)
        pymol.cmd.hide("everything", object)
        cmd.show("spheres", object)
        pymol.cmd.set("sphere_scale", 0.25)
        pymol.cmd.set("sphere_transparency", 0)
        pymol.cmd.set_color(f"custom_color_{index+10000}", list(colors_1[index][:3]))
        # pymol.cmd.set_color(f"custom_color_{index+10000}", color_0)
        pymol.cmd.color(f"custom_color_{index+10000}", object)

    print('done')

def color_h(selection='all'):
    s = str(selection)
    # print(s)
    cmd.set_color('color_ile',[0.996,0.062,0.062])
    cmd.set_color('color_phe',[0.996,0.109,0.109])
    cmd.set_color('color_val',[0.992,0.156,0.156])
    cmd.set_color('color_leu',[0.992,0.207,0.207])
    cmd.set_color('color_trp',[0.992,0.254,0.254])
    cmd.set_color('color_met',[0.988,0.301,0.301])
    cmd.set_color('color_ala',[0.988,0.348,0.348])
    cmd.set_color('color_gly',[0.984,0.394,0.394])
    cmd.set_color('color_cys',[0.984,0.445,0.445])
    cmd.set_color('color_tyr',[0.984,0.492,0.492])
    cmd.set_color('color_pro',[0.980,0.539,0.539])
    cmd.set_color('color_thr',[0.980,0.586,0.586])
    cmd.set_color('color_ser',[0.980,0.637,0.637])
    cmd.set_color('color_his',[0.977,0.684,0.684])
    cmd.set_color('color_glu',[0.977,0.730,0.730])
    cmd.set_color('color_asn',[0.973,0.777,0.777])
    cmd.set_color('color_gln',[0.973,0.824,0.824])
    cmd.set_color('color_asp',[0.973,0.875,0.875])
    cmd.set_color('color_lys',[0.899,0.922,0.922])
    cmd.set_color('color_arg',[0.899,0.969,0.969])
    cmd.color("color_ile","("+s+" and resn ile)")
    cmd.color("color_phe","("+s+" and resn phe)")
    cmd.color("color_val","("+s+" and resn val)")
    cmd.color("color_leu","("+s+" and resn leu)")
    cmd.color("color_trp","("+s+" and resn trp)")
    cmd.color("color_met","("+s+" and resn met)")
    cmd.color("color_ala","("+s+" and resn ala)")
    cmd.color("color_gly","("+s+" and resn gly)")
    cmd.color("color_cys","("+s+" and resn cys)")
    cmd.color("color_tyr","("+s+" and resn tyr)")
    cmd.color("color_pro","("+s+" and resn pro)")
    cmd.color("color_thr","("+s+" and resn thr)")
    cmd.color("color_ser","("+s+" and resn ser)")
    cmd.color("color_his","("+s+" and resn his)")
    cmd.color("color_glu","("+s+" and resn glu)")
    cmd.color("color_asn","("+s+" and resn asn)")
    cmd.color("color_gln","("+s+" and resn gln)")
    cmd.color("color_asp","("+s+" and resn asp)")
    cmd.color("color_lys","("+s+" and resn lys)")
    cmd.color("color_arg","("+s+" and resn arg)")
    cmd.extend('color_h',color_h)

def distance_h(object = "all", sur = 'all'):
    s = str(object)
    d_ile = cmd.distance('d' + s + "_ile", s, "(resn ile)", cutoff = 4)
    d_phe = cmd.distance('d' + s + "_phe", s, "(resn phe)", cutoff = 4)
    d_val = cmd.distance('d' + s + "_val", s, "(resn val)", cutoff = 4)
    d_leu = cmd.distance('d' + s + "_leu", s, "(resn leu)", cutoff = 4)
    d_trp = cmd.distance('d' + s + "_trp", s, "(resn trp)", cutoff = 4)
    d_met = cmd.distance('d' + s + "_met", s, "(resn met)", cutoff = 4)
    d_ala = cmd.distance('d' + s + "_ala", s, "(resn ala)", cutoff = 4)
    d_gly = cmd.distance('d' + s + "_gly", s, "(resn gly)", cutoff = 4)
    d_cys = cmd.distance('d' + s + "_cys", s, "(resn cys)", cutoff = 4)
    d_pro = cmd.distance('d' + s + "_pro", s, "(resn pro)", cutoff = 4)
    d_thr = cmd.distance('d' + s + "_thr", s, "(resn thr)", cutoff = 4)
    d_ser = cmd.distance('d' + s + "_ser", s, "(resn ser)", cutoff = 4)
    d_his = cmd.distance('d' + s + "_his", s, "(resn his)", cutoff = 4)
    d_asn = cmd.distance('d' + s + "_asn", s, "(resn asn)", cutoff = 4)
    d_gln = cmd.distance('d' + s + "_gln", s, "(resn gln)", cutoff = 4)
    d_asp = cmd.distance('d' + s + "_asp", s, "(resn asp)", cutoff = 4)
    d_lys = cmd.distance('d' + s + "_lys", s, "(resn lys)", cutoff = 4)
    d_arg = cmd.distance('d' + s + "_arg", s, "(resn arg)", cutoff = 4)

    distances = [
        ('d' + s + '_ile', d_ile), ('d' + s + '_phe', d_phe), ('d' + s + '_val', d_val), ('d' + s + '_leu', d_leu),
        ('d' + s + '_trp', d_trp), ('d' + s + '_met', d_met), ('d' + s + '_ala', d_ala), ('d' + s + '_gly', d_gly),
        ('d' + s + '_cys', d_cys), ('d' + s + '_pro', d_pro), ('d' + s + '_thr', d_thr), ('d' + s + '_ser', d_ser),
        ('d' + s + '_his', d_his), ('d' + s + '_asn', d_asn), ('d' + s + '_gln', d_gln), ('d' + s + '_asp', d_asp),
        ('d' + s + '_lys', d_lys), ('d' + s + '_arg', d_arg)
    ]

    output_distances = []

    for distance in distances:
        if distance[1] <= 0:
            pymol.cmd.delete(distance[0])
        else:
            output_distances.append(distance)

    cmd.extend('distance_h', distance_h)

    return output_distances


def color_h2(selection='all'):
    s = str(selection)
    print(s)
    cmd.set_color("color_ile2",[0.938,1,0.938])
    cmd.set_color("color_phe2",[0.891,1,0.891])
    cmd.set_color("color_val2",[0.844,1,0.844])
    cmd.set_color("color_leu2",[0.793,1,0.793])
    cmd.set_color("color_trp2",[0.746,1,0.746])
    cmd.set_color("color_met2",[0.699,1,0.699])
    cmd.set_color("color_ala2",[0.652,1,0.652])
    cmd.set_color("color_gly2",[0.606,1,0.606])
    cmd.set_color("color_cys2",[0.555,1,0.555])
    cmd.set_color("color_tyr2",[0.508,1,0.508])
    cmd.set_color("color_pro2",[0.461,1,0.461])
    cmd.set_color("color_thr2",[0.414,1,0.414])
    cmd.set_color("color_ser2",[0.363,1,0.363])
    cmd.set_color("color_his2",[0.316,1,0.316])
    cmd.set_color("color_glu2",[0.27,1,0.27])
    cmd.set_color("color_asn2",[0.223,1,0.223])
    cmd.set_color("color_gln2",[0.176,1,0.176])
    cmd.set_color("color_asp2",[0.125,1,0.125])
    cmd.set_color("color_lys2",[0.078,1,0.078])
    cmd.set_color("color_arg2",[0.031,1,0.031])
    cmd.color("color_ile2","("+s+" and resn ile)")
    cmd.color("color_phe2","("+s+" and resn phe)")
    cmd.color("color_val2","("+s+" and resn val)")
    cmd.color("color_leu2","("+s+" and resn leu)")
    cmd.color("color_trp2","("+s+" and resn trp)")
    cmd.color("color_met2","("+s+" and resn met)")
    cmd.color("color_ala2","("+s+" and resn ala)")
    cmd.color("color_gly2","("+s+" and resn gly)")
    cmd.color("color_cys2","("+s+" and resn cys)")
    cmd.color("color_tyr2","("+s+" and resn tyr)")
    cmd.color("color_pro2","("+s+" and resn pro)")
    cmd.color("color_thr2","("+s+" and resn thr)")
    cmd.color("color_ser2","("+s+" and resn ser)")
    cmd.color("color_his2","("+s+" and resn his)")
    cmd.color("color_glu2","("+s+" and resn glu)")
    cmd.color("color_asn2","("+s+" and resn asn)")
    cmd.color("color_gln2","("+s+" and resn gln)")
    cmd.color("color_asp2","("+s+" and resn asp)")
    cmd.color("color_lys2","("+s+" and resn lys)")
    cmd.color("color_arg2","("+s+" and resn arg)")
    cmd.extend('color_h2',color_h2)

def com(selection, state=None, mass=None, object=None, quiet=1, **kwargs):
    quiet = int(quiet)
    if (object == None):
        try:
            object = cmd.get_legal_name(selection)
            object = cmd.get_unused_name(object + "_COM", 0)
        except AttributeError:
            object = 'COM'
    cmd.delete(object)

    if (state != None):
        x, y, z = get_com(selection, mass=mass, quiet=quiet)
        if not quiet:
            print("%f %f %f" % (x, y, z))
        cmd.pseudoatom(object, pos=[x, y, z], **kwargs)
        cmd.show("spheres", object)
    else:
        for i in range(cmd.count_states()):
            x, y, z = get_com(selection, mass=mass, state=i + 1, quiet=quiet)
            if not quiet:
                print("State %d:%f %f %f" % (i + 1, x, y, z))
            cmd.pseudoatom(object, pos=[x, y, z], state=i + 1, **kwargs)
            cmd.show("spheres", 'last ' + object)

cmd.extend("com", com)


def get_com(selection, state=1, mass=None, quiet=1):
    """
 DESCRIPTION

    Calculates the center of mass

    Author: Sean Law
    Michigan State University
    slaw (at) msu . edu
    """
    quiet = int(quiet)

    totmass = 0.0
    if mass != None and not quiet:
        print("Calculating mass-weighted COM")

    state = int(state)
    model = cmd.get_model(selection, state)
    x, y, z = 0, 0, 0
    for a in model.atom:
        if (mass != None):
            m = a.get_mass()
            x += a.coord[0] * m
            y += a.coord[1] * m
            z += a.coord[2] * m
            totmass += m
        else:
            x += a.coord[0]
            y += a.coord[1]
            z += a.coord[2]

    if (mass != None):
        return x / totmass, y / totmass, z / totmass
    else:
        return x / len(model.atom), y / len(model.atom), z / len(model.atom)

cmd.extend("get_com", get_com)