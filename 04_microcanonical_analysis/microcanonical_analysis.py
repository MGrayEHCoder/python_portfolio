import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from scipy.special import logsumexp
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, QhullError
from matplotlib.colors import ListedColormap

plt.rcParams['text.usetex'] = False

# Project Paths

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)



def recurs(E, beta, logsumh, Mlog):
    glog = 0 * E
    glog_prev = 0 * E + 1

    Zlog = np.zeros(len(beta)) + 1

    while maxDev(glog, glog_prev) > 0.05:

        for _ in range(10):

            glog_prev = np.copy(glog)

            for n in range(len(E)):
                glog[n] = (
                    logsumh[n]
                    - logsumexp(
                        Mlog - beta * E[n] - Zlog
                    )
                )

            for i in range(len(beta)):
                Zlog[i] = logsumexp(
                    glog - beta[i] * E
                )

    return E, glog

def histReweight(folder,nearestBend,temprange=[0,5],binrange=[2,10]):      
    temps = pd.read_csv(folder+'threads.csv')
    filt1=(temps.loc[:,'Temp']>=temprange[0]) & (temps.loc[:,'Temp']<=temprange[1]) & (temps.loc[:,'BendingStrength']==nearestBend)
    data=pd.concat((pd.read_csv(folder+'EHist'+str(i)+'.csv').assign(Temp = temps.loc[i,'Temp'],Thread = i) for i in temps.loc[filt1,'id']), ignore_index=True)
    filt2=(data.loc[:,'Bin']>=binrange[0]) & (data.loc[:,'Bin']<=binrange[1])
    dataStab=data[filt2]
    summed = dataStab.groupby(['Temp','EBinEnergy'])['N'].sum()
    h=summed.unstack(level=0)
    M=h.sum() #all the same (one for each temp)
    Mlog=np.log(h.sum()) #all the same (one for each temp)
    hNorm=(h/M) #normalized hist (Array for each temp)
    Ones=M*0+1
    beta=Ones*1/h.columns # (one for each temp)

    hsum = h.sum(axis=1) #one for each energy bin
    # logsumh=np.log(hsum) #one for each energy bin
    logsumh = np.zeros_like(hsum, dtype=float)
    logsumh[hsum == 0] = -np.inf
    logsumh[hsum != 0] = np.log(hsum[hsum != 0])
    OnesE=hsum*0+1  #one for each energy
    E=h.index.values*OnesE #one for each energy

    return recurs(E.to_numpy(),beta.to_numpy(),logsumh,Mlog.to_numpy())

def histReweight(
    folder,
    nearestBend,
    temprange=(0, 5),
    binrange=(2, 10)
):
    folder = Path(folder)

    temps = pd.read_csv(
        folder / "threads.csv"
    )

    filt1 = (
        (temps["Temp"] >= temprange[0])
        & (temps["Temp"] <= temprange[1])
        & (temps["BendingStrength"] == nearestBend)
    )

    data = pd.concat(
        (
            pd.read_csv(
                folder / f"EHist{i}.csv"
            ).assign(
                Temp=temps.loc[i, "Temp"],
                Thread=i
            )
            for i in temps.loc[filt1, "id"]
        ),
        ignore_index=True
    )

    filt2 = (
        (data["Bin"] >= binrange[0])
        & (data["Bin"] <= binrange[1])
    )

    data_stab = data[filt2]

    summed = (
        data_stab
        .groupby(["Temp", "EBinEnergy"])["N"]
        .sum()
    )

    h = summed.unstack(level=0)

    M = h.sum() #all the same (one for each temp)

    Mlog = np.log(h.sum()) #all the same (one for each temp)

    h_norm = h / M # normalized histogram array

    ones = M * 0 + 1

    beta = ones * 1 / h.columns # (one for each temp)

    hsum = h.sum(axis=1)

    logsumh = np.zeros_like(
        hsum,
        dtype=float
    )

    logsumh[hsum == 0] = -np.inf

    logsumh[hsum != 0] = np.log(
        hsum[hsum != 0]
    )

    ones_E = hsum * 0 + 1

    E = h.index.values * ones_E

    return recurs(
        E.to_numpy(),
        beta.to_numpy(),
        logsumh,
        Mlog.to_numpy()
    )

def maxDev(glog, glog_prev):
    is_finite = np.logical_and(
        np.isfinite(glog),
        np.isfinite(glog_prev)
    )

    diff = glog[is_finite] - glog_prev[is_finite]

    return np.abs(diff).max()

def PlotPhaseEnergies(folder, C, ax, fig, Shapes, length):

    percentCut = 0.5

    folder = Path(folder)

    temps = pd.read_csv(
        folder / "threads.csv"
    )

    ax.set_xlabel("Energy")
    ax.set_ylabel(r"$S_{bend}$")

    phaseShapes = Shapes

    for bend in temps["BendingStrength"].unique():

        bendnum = float(bend)

        E, glog = histReweight(
            folder,
            bend
        )

        Estep = E[1] - E[0]

        # Keep only finite values
        finite_mask = np.isfinite(glog)

        S = glog[finite_mask]
        E_0 = E[finite_mask]


        # Store phase-transition energies found
        # at the previous derivative order.
        prevE_Phase = []


        # Examine derivatives of orders 2, 3, and 4
        for j in range(3):

            derivative_order = j + 2

            w = findWindowLength(
                S,
                derivative_order,
                Estep,
                0.8
            )

            derivative = savgol_filter(
                S,
                window_length=w,
                polyorder=derivative_order,
                deriv=derivative_order,
                delta=Estep
            )


            # Remove the upper portion of the energy range
            # where the derivative is not being considered.
            derivative = derivative[
                :-int(len(derivative) * percentCut)
            ]


            # Find sign changes in the derivative.
            derivative_product = (
                derivative[1:]
                * derivative[:-1]
            )

            zero_crossing_values = [
                value
                for value in derivative_product
                if value < 0
            ]

            # Keep every other crossing.
            zero_crossing_values = [
                value
                for i, value in enumerate(
                    zero_crossing_values
                )
                if i % 2 == 0
            ]


            # Find the corresponding indices.
            indices = []

            for value in zero_crossing_values:

                index = np.where(
                    derivative_product == value
                )

                indices.append(
                    int(index[0][0])
                )


            rawE_phase = E_0[indices]


            # ====================================================
            # Remove Higher-Order Repeats
            # ====================================================

            E_phase = []


            if len(prevE_Phase) > 0:

                repeat = False

                for rawE in rawE_phase:

                    for prevE in prevE_Phase:

                        diff = abs(
                            rawE - prevE
                        )

                        if diff <= 10:
                            repeat = True

                    if repeat is False:
                        E_phase.append(rawE)


            else:

                repeat = False

                for i, rawE in enumerate(rawE_phase):

                    for otherrawE in np.flip(
                        rawE_phase
                    )[:-i]:

                        if otherrawE != rawE:

                            diff = abs(
                                rawE - otherrawE
                            )

                            if diff <= 5:
                                repeat = True

                    if repeat is False:
                        E_phase.append(rawE)


            # ====================================================
            # Plot Transitions
            # ====================================================

            if len(E_phase) > 0:

                for energy in E_phase:

                    ax.plot(
                        energy,
                        bendnum,
                        marker=phaseShapes[j],
                        markersize=10,
                        markeredgecolor="black",
                        markerfacecolor="black"
                    )


            # IMPORTANT:
            # Store the RAW transitions, not the filtered ones.
            prevE_Phase = rawE_phase


    # ========================================================
    # Save Phase Diagram
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / f"L{length}_Phase_Diagram_"
          f"ConstBend_{C}.png"
    )

    fig.savefig(
        output_file,
        dpi=650
    )

def findWindowLength(S, order, Estep, percentCut):
    """
    Determine the Savitzky-Golay window length by looking
    for a plateau in the number of detected transitions.
    """

    num_of_transitions = []

    for k in range(6, 200):

        derivative = savgol_filter(
            S,
            window_length=k,
            polyorder=order,
            deriv=order,
            delta=Estep
        )

        derivative = derivative[
            :int(percentCut * len(derivative))
        ]

        derivative_product = (
            derivative[1:]
            * derivative[:-1]
        )

        zero_crossings = [
            value
            for value in derivative_product
            if value < 0
        ]

        zero_crossings = [
            value
            for i, value in enumerate(zero_crossings)
            if i % 2 == 0 and i < 450
        ]

        num_of_transitions.append(
            len(zero_crossings)
        )


    # Number of consecutive equal values required
    # to identify a plateau.
    plateau_size = 5

    prev_value = 0
    count = 0
    window_length = 0
    plateau_found = False

    i = 0

    while not plateau_found:

        if num_of_transitions[i] == prev_value:

            count += 1
            window_length = i

            if count == plateau_size:
                plateau_found = True
                window_length = i - count

        else:
            count = 0

        prev_value = num_of_transitions[i]
        i += 1

        if i == len(num_of_transitions):
            plateau_found = True
            window_length = len(num_of_transitions) - 3

    return window_length



def make_legend(shapes):

    fig, ax = plt.subplots(dpi=450)

    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0.75, 3)

    for i, shape in enumerate(shapes):

        message = (
            f"= {i + 1} Order Phase Transition"
        )

        ax.plot(
            1,
            i,
            marker=shape,
            markersize=12,
            markeredgecolor="black",
            markerfacecolor="black"
        )

        ax.text(
            1.15,
            i - 0.05,
            message,
            fontsize=12
        )

    fig.savefig(
        OUTPUT_DIR / "legend.png"
    )

    return fig


def make_folders(L):

    folders = []

    CBs = [
        "0.5",
        "0.625",
        "0.75",
        "0.875",
        "1.0"
    ]

    if L == 40:
        date_string = "24-06-28"
    else:
        date_string = "24-05-14"

    for Cb in CBs:

        folder = (
            DATA_DIR
            / str(L)
            / f"{date_string}_{L}_Constbend {Cb}"
            / "data"
        )

        folders.append(folder)

    return folders

def MakePhaseRegions(length,width,points_on = True):

    # Load data"C:\Users\gbrub\OneDrive - murraystate.edu\Research\SavedStructureData_Labels40.csv"
    structure_data_file = (
        DATA_DIR / f"SavedStructureData_Labels{length}.csv"
    )

    all_contacts_with_label = pd.read_csv(
        structure_data_file
    )

    # Replace labels for consistency
    all_contacts_with_label['Label'] = all_contacts_with_label['Label'].replace(
        {'2S_Globule': 'Globule', 
        '2S_ArcJoint': '2S', 
        'GlobuleRing': 'Ring'})

    # Define unique bending strengths and energy bins
    bending_strength_values = sorted(all_contacts_with_label['BendingStrength'].unique())
    d_bending_strength = bending_strength_values[1] - bending_strength_values[0]
    bending_strength_values = bending_strength_values - 0.5 * d_bending_strength
    bending_strength_bins = np.append(bending_strength_values, bending_strength_values[-1] + d_bending_strength)

    energy_min = all_contacts_with_label['Energy'].min()
    energy_max = all_contacts_with_label['Energy'].max()
    energy_bins = np.linspace(energy_min, energy_max, 100)

    # Create copies of data for expansion
    all_contacts_with_label_high_bs = all_contacts_with_label.copy()
    all_contacts_with_label_low_bs = all_contacts_with_label.copy()
    all_contacts_with_label_high_bs['BendingStrength'] += d_bending_strength * 0.35
    all_contacts_with_label_low_bs['BendingStrength'] -= d_bending_strength * 0.35

    all_contacts_with_label_high_e = all_contacts_with_label.copy()
    all_contacts_with_label_low_e = all_contacts_with_label.copy()
    all_contacts_with_label_high_e['Energy'] += 2
    all_contacts_with_label_low_e['Energy'] -= 2

    # Concatenate expanded data
    all_contacts_with_label_exp = pd.concat(
        [all_contacts_with_label_high_bs, all_contacts_with_label_low_bs,
        all_contacts_with_label_high_e, all_contacts_with_label_low_e],
        ignore_index=True
    )

    structure_names = sorted(all_contacts_with_label['Label'].unique())

    #Define colormap
    preset_cmap = plt.get_cmap('tab20', len(structure_names))
    cmap = ListedColormap(preset_cmap.colors)


    # Plotting
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot Convex Hulls
    for i, structure in enumerate(structure_names):
        to_plot_exp = all_contacts_with_label_exp[
            (all_contacts_with_label_exp['Label'] == structure) &
            (all_contacts_with_label_exp['PotWidth'] == width)
        ]   
    
        points_exp = np.array([to_plot_exp['Energy'], to_plot_exp['BendingStrength']]).transpose()

        # Convex Hull
        if len(points_exp) > 2:
            try:
                hull = ConvexHull(points_exp)
                ax.fill(points_exp[hull.vertices, 0], points_exp[hull.vertices, 1], color=cmap(i), alpha=0.2, label=structure)
            except QhullError:
                print(f"Convex hull could not be computed for structure {structure} due to collinear points.")

    # Plot points on top of Convex Hulls
    if points_on:
        for i, structure in enumerate(structure_names):
            to_plot_orig = all_contacts_with_label[
                (all_contacts_with_label['Label'] == structure) &
                (all_contacts_with_label['PotWidth'] == width)
            ]
        
            points_orig = np.array([to_plot_orig['Energy'], to_plot_orig['BendingStrength']]).transpose()

            if len(points_orig) > 0:
                ax.scatter(points_orig[:, 0], points_orig[:, 1], color=cmap(i), s=8)

    # Customize plot
    ax.set_ylabel('Bending Strength')
    ax.set_xlabel('Energy')
    ax.set_xlim([-40,125])
    ax.set_ylim([7,20])
    ax.set_title(f'n = {length}    w = {width}')
    plt.legend()
    return fig, ax

def MakePhaseDiagram(length):

    folders = make_folders(length)

    CBs = [
        "0.5",
        "0.625",
        "0.75",
        "0.875",
        "1.0"
    ]

    shapes = [
        "o",
        "s",
        "d",
        "^"
    ]

    make_legend(shapes)

    for i in range(len(folders)):

        fig, ax = MakePhaseRegions(
            length,
            float(CBs[i]),
            False
        )

        PlotPhaseEnergies(
            folders[i],
            CBs[i],
            ax,
            fig,
            shapes,
            length
        )

        plt.close(fig)

def MakeMicrocanonicalDataPlots(
    folder,
    length,
    width,
    BS
):

    fig, axs = plt.subplots(
        5,
        1,
        figsize=(8, 12)
    )

    percentCut = 0.8

    folder = Path(folder)

    temps = pd.read_csv(
        folder / "threads.csv"
    )

    BS = temps[
        "BendingStrength"
    ].unique()[BS]

    E, glog = histReweight(
        folder,
        BS
    )

    Estep = E[1] - E[0]

    finite_mask = np.isfinite(glog)

    S = glog[finite_mask]
    E_0 = E[finite_mask]


    # --------------------------------------------------------
    # Microcanonical entropy
    # --------------------------------------------------------

    cutoff_index = int(
        len(E_0) * percentCut
    )

    axs[0].plot(
        E_0[:cutoff_index],
        S[:cutoff_index]
    )

    axs[0].set_xticks([])

    axs[0].set_xlim(-40, 125)
    axs[0].set_ylim(-100, 75)


    # --------------------------------------------------------
    # Determine derivative windows
    # --------------------------------------------------------

    window_lengths = []

    for j in range(4):

        window_lengths.append(
            findWindowLength(
                S,
                j + 1,
                Estep,
                0.8
            )
        )

    # Preserve the original override.
    window_lengths[0] = 10


    # --------------------------------------------------------
    # Plot derivatives
    # --------------------------------------------------------

    derivative_limits = [
        (-1, 7),
        (-0.4, 0.1),
        (-0.04, 0.025),
        (-0.002, 0.002)
    ]

    for j in range(4):

        axs[j + 1].set_xlim(
            -40,
            125
        )

        axs[j + 1].axhline(
            y=0,
            color="red",
            linestyle="--",
            linewidth=0.5
        )

        derivative = savgol_filter(
            S,
            window_length=window_lengths[j],
            polyorder=j + 1,
            deriv=j + 1,
            delta=Estep
        )

        derivative = derivative[:cutoff_index]

        axs[j + 1].plot(
            E_0[:len(derivative)],
            derivative
        )

        axs[j + 1].set_ylim(
            *derivative_limits[j]
        )

        if j < 3:
            axs[j + 1].set_xticks([])


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"MicroCanonicalDataPlot_"
          f"L{length}_w{width}_"
          f"BendingStrength{BS}.png"
    )

    fig.savefig(
        output_file,
        dpi=650
    )

    plt.close(fig)
    
def main():
    MakePhaseDiagram(40)


if __name__ == "__main__":
    main()
