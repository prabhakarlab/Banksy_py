'''
This script contains all the utility objects (e.g., dictionaries for padding clusters etc)
for the Slideseq dataset.

NOTE that for data analysis we recommend users to define their own references.
'''

# labels for clusters in dropviz scRNAseq dataset (see dropviz.org)
dropviz_dict = {
    "GranularNeuron_Gabra6": 1,
    "PurkinjeNeuron_Pcp2": 2,
    "Interneurons_Pvalb": 3,
    "Interneurons_and_Other_Nnat": 4,
    "Microglia_Macrophage_C1qb": 5,
    "Oligodendrocyte_Polydendrocyte_Tfr_Tnr": 6,
    "BergmannGlia_Gpr37l1": 7,
    "Astrocyte_Gja1": 8,
    "Choroid_Plexus_Ttr": 9,
    "Endothelial_Flt1": 10,
    "Fibroblast-Like_Dcn": 11,
}

# Slide dropviz reference DE genes
markergenes_dict = {
    "GranularNeuron_Gabra6": [
        "Sh3bgr", "Fat2", "Kcnj12", "Bcl2l15", "Reln", "Uncx", "Plk5", "Tmem163", "Grin2a", "Rims3", "Lrrtm3", "Mctp1",
        "Grin2c", "Ipcef1", "Nr4a2", "Sptb"
    ],
    "PurkinjeNeuron_Pcp2": [
        "Arhgef33", "Ppp1r17", "Nrk", "Clmn", "Slc1a6", "Atp2a3", "Nexn", "Cck", "Camk2a", "Ppp1r16b", "Sycp1", "Itpka",
        "RP23-372N19.2", "Trabd2b", "Gm5083", "Cabp7", "Gm14033", "Cacna1g", "Prkcg", "B3gnt5", "Scn4b", "Cep76",
        "Pde5a", "Shank2", "Kcna6", "Tmem255a", "Prmt8", "Fam174b", "Eps8l2", "Stk17b", "Cmya5", "Plxdc1", "Col18a1",
        "Hes3", "Cntnap5b", "Krt25", "Scd3", "Trpc3", "Atrnl1", "Iltifb", "Kcng4", "Tuba8", "A330050F15Rik", "Shisa6",
        "Bcl11a", "Dagla", "Htr1b", "Nell1", "Ppp4r4", "Cacnb2", "Ebf2", "Zfp385c", "Clip4", "Srgap1", "Arhgap32",
        "Plekhd1", "Grik1", "Epha5", "Kcnip1", "Slc35f1", "Car7", "Ptprm", "Large", "Pcsk6", "Robo2", "Adgrl2", "Lhx5",
        "Ptchd4", "Casq2"
    ],
    "Interneurons_Pvalb": [
        "Kit", "Sorcs3", "Gria3", "Tfap2b", "Slc24a3", "B3galt2", "Prkcd", "Whrn", "Ntn1", "Grik3", "Lypd6", "Ar",
        "Socs2", "Penk", "Nrsn2", "Parm1", "Sla", "Grid1", "N28178", "Nav1", "Plch1", "Esrrg", "Frmpd4", "Hpca",
        "Asic2", "Trim67", "Pak3", "Nxph1", "Vstm2l", "Grin2b", "Nos1ap", "Slc32a1", "Adarb2", "Sema3e", "Skor1",
        "Thsd7a", "Lamb1", "Mar-11", "Neurod6", "Cacna2d3", "Hunk", "Megf10", "Plxnc1", "Ret", "Tmem132e", "Cntn4",
        "A830039N20Rik", "Kcnab3", "Steap2", "Ajap1", "Chst1", "Dusp10", "Mgat5", "Epha8", "Hcn1", "Kcng4", "Psd2",
        "Chd5", "Cttnbp2", "Fam84a", "Grin3a", "Phactr2", "Shisa6", "Btbd11", "Gdpd5", "Prdm8", "Shank2", "Pam",
        "Arhgap32", "Csrnp3", "Sipa1l3", "AI504432", "Galnt18", "Lhfpl3", "Prkca", "Xkr4", "2310067B10Rik"
    ],
    "Interneurons_and_Other_Nnat": [
        "Sln", "Sst", "Nrgn", "Slc17a6", "Doc2g", "Dclk3", "Grm2", "Rgs6", "Vgf", "Grm5", "Tpbg", "Lypd1", "Adcy2",
        "Tcf7l2", "Slc6a5", "Fam19a2", "Neurod6", "Samd3", "Zcchc12", "Pax2", "Basp1", "Rbms3", "Edil3", "Rab3b",
        "Nxph4", "Rasgrp1"
    ],
    "Microglia_Macrophage_C1qb": [
        'C1qb', 'Ctss', 'C1qc', 'C1qa', 'Cx3cr1', 'Csf1r', 'Tyrobp', 'Laptm5', 'Siglech', 'Fcer1g',
        'Trem2', 'Lyz2', 'Fcrls', 'P2ry12', 'Aif1', 'Mrc1', 'Ly86', 'Stab1', 'Selplg', 'Cd74', 'Gpr34',
        'Pf4', 'Fcgr3', 'Rnase4', 'Olfml3', 'Cd68', 'Unc93b1', 'Ctsc', 'Mpeg1', 'Rgs10', 'H2-Ab1',
        'F13a1', 'H2-Eb1',
    ],
    "Oligodendrocyte_Polydendrocyte_Tfr_Tnr": [
        "Mag", "Cldn11", "Mal", "Mog", "Ermn", "Olig1", "Ugt8a", "Opalin", "Gjb1", "Pllp", "Tmem88b", "Gpr37",
        "Gm21984", "Serpinb1a", "Tspan2", "Klk6", "Aspa", "Efnb3", "Sox2ot", "Ppp1r14a", "Gsn", "Pex5l", "Anln", "Il33",
        "Litaf", "Edil3", "Hapln2", "Tnfaip6", "Pdlim2", "Sox10", "Car14", "Fa2h", "Gjc2", "Nkx6-2", "Tmem125",
        "2810468N07Rik", "Arsg", "Prr5l", "Trim59", "Olig2", "Tmem63a", "Adamts4", "Tmeff1", "Grb14", "Lpar1",
        "Carhsp1", "Evi2a", "Gstm7", "Pls1", "Rhog", "Gltp", "Psat1", "Galnt6", "Enpp6", "Cd82", "Cers2", "Elovl7",
        "Pcolce2", "Plcl1", "Plekhh1", "Tmem98", "Tmcc3", "Adssl1", "Il1rap", "Itgb4", "Bcas1", "Frmd8",
        "RP23-240E15.3", "Rtkn", "Tmbim1", "Gal3st1", "Insc", "Prr18", "Jam3", "Cpm", "Myrf", "Erbb2ip", "Fgfr2",
        "Enpp4", "Kctd4", "Sh3gl3", "Neat1", "Plin3", "Elovl1", "Ppfibp2"
    ],
    "BergmannGlia_Gpr37l1": [
        "Gdf10", "Id4", "Tlcd1", "Cacng5", "Npy", "Lcat", "Btbd17", "Lgi4", "Kcnj16", "Cyp2j9", "Slc14a1", "Gas1",
        "Cml1", "Luzp2", "Itih3", "Tubb2b", "Hopx", "Cldn10", "Elovl2", "Tnc", "Gpc6", "Mybpc1", "Cthrc1", "Dao", "A2m",
        "Ddah1", "Col9a3", "Sox2", "Cdc42ep4", "Gjc3", "Cpne2", "Fzd1", "Prex1", "Ctxn3", "Gpld1", "Pax3", "Slc7a10",
        "Abi3bp", "Lfng", "Rab34", "Slc25a18", "Cd70", "Paqr6", "F3", "Mgst1", "Cdc42ep1", "Cxcl14", "Msx2", "Shisa9",
        "Omg", "Aldh1l1", "Dpy19l3", "Proca1", "Npas3", "1700084C01Rik", "3110082J24Rik", "Slc12a4", "Slco4a1",
        "Tspan15", "Dbx2", "Fxyd7", "Gnb4", "Pdlim4", "Rftn2", "Cd302", "Prex2", "Smpdl3a", "Bok", "E130114P18Rik",
        "Fam20a", "Gucy1a3", "Mertk", "S100a6", "Sox1", "Syt10", "Tmem198b", "Pmp22", "Scrg1", "0610040J01Rik",
        "AI464131", "Gabrg1", "Hk2", "Lrrc2", "Plekho2", "Rasl11a", "Trib2", "Btd", "Cables1", "Irx5", "Jam2", "Lgr6",
        "Mob3b", "Slc39a12", "Stk32a", "Tom1l1", "Abhd3", "Fgfr3", "Pou3f3", "Etnppl", "Marcksl1", "Gabra4", "Slc27a1",
        "Efnb2", "2310022B05Rik", "Emp2", "Gng12", "Itgb8", "Laptm4b", "Tmem176b", "Adamts1", "Phgdh", "Rarres2",
        "Srebf1", "Cib1", "Pcdh10", "Rdh5", "Tst", "Adgra1", "Efhd1", "Pigs", "Rorb", "Cyp2d22", "Fscn1", "Gng5",
        "Hdac8", "Klhl13", "Lbh", "Mboat2", "Nfe2l2", "Pbxip1", "Sash1", "Acad8", "Eva1a", "Fgfr2", "Hist1h1c", "Naaa",
        "Rgma", "Soat1", "Sod3", "Ctso", "Dusp6", "Emid1", "Klf3", "Notch1", "Pantr1", "Pcdh17", "Rnaset2a", "Slc13a3",
        "Smox", "Srgap1", "St3gal6", "Abhd4", "Ccdc24", "Cpq", "Nebl", "Pou3f2", "Rab31", "Rhoc", "Rhoj", "Sox5",
        "Zfyve21"
    ],
    "Astrocyte_Gja1": [
        "Hhatl", "Cd38", "Cd44", "Cyp26b1", "Efemp1", "Slc6a11", "Gfap", "Epha4", "Adgrg1", "Eva1a"
    ],
    "Choroid_Plexus_Ttr": [
        'Ccdc153', '1500015O10Rik', 'Gm973', 'Rsph1', 'Mia', 'Tmem212', 'Enkur', 'Dynlrb2',
        'Igfbp2', 'Fam183b', 'Ccdc146', 'Prlr', 'Sostdc1', 'Foxj1', 'Cfap126', 'Sntn', 'Kl',
        'F5', 'Kcnj13', 'Ak7', 'Calml4', 'Wdr66', 'Iqca', '1700016K19Rik', 'Folr1', 'Fhad1',
        'Clic6', 'Rbp1', 'Lrriq1', 'Meig1', 'Cfap45', 'Efcab10', 'Rsph4a', 'Mns1', 'Ezr',
        'Efcab1', 'Kcne2', 'Ccnd2', 'Col8a1', 'Tmem72', 'Car12', 'Cdkn1c', 'Prr32',
    ],
    "Endothelial_Flt1": [
        'Ly6c1', 'Ly6a', 'Cldn5', 'Slco1c1', 'Itm2a', 'Slco1a4', 'Flt1', 'Adgrf5', 'Abcb1a', 'Klf2',
        'Cxcl12', 'Srgn', 'Ctla2a', 'Sdpr', 'Ptprb', 'Palmd', 'Adgrl4', 'Ramp2', 'Wfdc1', 'Emcn',
        'Cd34', 'Egfl7', 'Ifitm3', 'Fxyd5', 'Pecam1', '9430020K01Rik', 'Abcg2', 'Nostrin',
        'Ltbp4', 'Kdr', 'Sema3c', 'Pglyrp1', 'Lef1', 'Lmo2', 'Nes', 'Id1', 'Eng', 'Tm4sf1',
        'Anxa3', 'Tek', 'Acvrl1', 'Ahnak', 'Esam', 'Cav1', 'Lsr', 'C130074G19Rik', 'Fn1',
        'Klf4', 'Uaca', 'Slc22a8', 'Erg', 'BC028528', 'Foxq1', 'Clic5', 'Tmem204', 'Tinagl1',
        'Ets1', 'Podxl', 'Cdh5', 'Ccdc141', 'Ctsh', 'Slc40a1', 'Nrp1', 'Grap', 'Rasgrp3',
        'Slfn5', 'Cd93', 'Cmtm8', 'Kank3', 'Mfsd2a', 'Fli1', 'Epas1', 'Hspb1', 'Gkn3', 'Anxa2',
        'Gimap6', 'Csrp2', 'Isg15', 'Icam2', 'Tmem252', 'Ptrf', 'Gbp7', 'Il10rb', 'Psmb8',
        'Slc38a5', 'Edn3', 'Slc39a8', 'Slc7a5', 'Sox17', 'Arpc1b', 'Degs2', 'Grasp', 'Lgals9',
        'Scarb1', 'Thbd', 'Apold1', 'Trim47', 'AW112010', 'Bst2', 'Fcgrt', 'Sp100', 'Tshz2',
        'AU021092', 'Arhgap29', 'Grrp1', 'Itga6', 'Pdgfb', 'Ifitm2', 'Bcam', 'Gbp3', 'Itga1',
        'Lims2', 'Prom1', 'Slc16a2', 'Tagln2', 'Ddc', 'Edn1', 'Fzd6', 'Rsad2', 'Tie1', 'Bvht',
        'Hmgcs2', 'Net1', 'Pde2a', 'Wwtr1', 'Alpl', 'Gm694', 'H2-T23', 'She', 'Slc16a4', 'Tbx3',
        'Tgm2', 'Rgs12', 'St3gal6', '8430408G22Rik', 'Cyyr1', 'Gata2', 'Ifih1', 'Mmrn2', 'Ndnf',
        'Paqr5', 'Serpinb9', 'Col4a2', 'Ceacam1', 'Cp', 'Elk3', 'Elovl7', 'Foxf2', 'Gimap1',
        'Ppfibp1', 'Ly6c2', 'Robo4', 'Xaf1', 'Afap1l1', 'Col4a1', 'Rasip1', 'Slc7a1', 'Clec14a',
        'Fnbp1l', 'Hspa12b', 'St8sia4', 'Tdrp', 'Tgfbr2', 'Tmem140', 'Fas', 'Gm9946', 'Hmcn1',
        'Ifit1', 'Iigp1', 'Mecom', 'Rtp4', 'Tjp1', 'Ushbp1', 'Abhd2', 'Ocln', 'Atp10a', 'Capg',
        'Eogt', 'F11r', 'Ifi35', 'Lcn2', 'Tspo', 'Unc45b', 'Abcc4', 'Kif26a', 'Pir', 'Slco2b1',
        'St6galnac2', 'Cdkn1c', 'Adcy4', 'Ecscr', 'Filip1', 'Gbp2', 'Heg1', 'Serpinb6b', 'Synm',
        'Thsd1', 'Xdh', 'Cav2', 'Isyna1', 'Rhoc', 'Cnn2', 'Klhl5', 'Prss23', 'Vwf', 'Hes1',
        'Lima1', 'Polk', 'Tmem123', 'Vat1', 'B4galt4', 'Lipa', 'Tns1', 'Txnip', 'Irf9',
        'Swap70', 'Cfh', 'Rassf3',
    ],
    "Fibroblast-Like_Dcn": [
        'Vtn', 'Rgs5', 'Acta2', 'Igf2', 'Myl9', 'Crip1', 'Myh11', 'Dcn', 'Tagln', 'Tpm2', 'Higd1b', 'Lmod1',
        'Mylk', 'Rgs4', 'Filip1l', 'Aebp1', 'Mgp', 'Kcnj8', 'Art3', 'Casq2', 'Ndufa4l2', 'Slc6a20a', 'Aspn',
        'Cox4i2', 'Igfbp2', 'Nupr1', 'Col3a1', 'Pdgfrb', 'Cspg4', 'P2ry14', 'Sod3', 'Ifitm1', 'Ptrf', 'Slc7a11',
        'Sncg', 'Bgn', 'Pcolce', 'Mustn1', 'Slc6a13', 'Gm13861', 'Rbpms', 'Myo1b', 'Abcc9', 'Ace2', 'Cfh', 'Cp',
        'Gja4', 'Ogn', 'Perp', 'S1pr3', 'Bcam', 'Spp1', 'Gpx8', 'Col1a1', 'S100a11', 'Itga1', 'Ifitm2', 'Crispld2',
        'Des', 'Atp13a5', 'Rbp4', 'Ano1', 'Arhgdib', 'Olfr558', 'Pln', 'Snhg18', 'Nr2f2',
        'Gpc3', 'Rcn3', 'Cnn2', 'Emp3', 'Nexn', 'Adap2', 'Cd248', 'Col4a1', 'Foxc1', 'Slc22a6', 'Tagln2',
        'Hspb1', 'Enpep', 'Fmod', 'Serping1', 'Cdkn1c', 'Heyl', 'Efemp1', 'Gjb2', 'Colec12', 'Phldb2',
        'Wls', 'Arhgap29', 'Col4a2', 'Il34', 'Nid1', 'Slc13a4', 'Aldh1a2', 'Lum', 'Pde1a', 'Tbx18', 'Tinagl1',
        'Flna', 'Serpinf1', 'Rbp1', 'Lamb2', 'Map3k7cl', 'C1qtnf2', 'Crabp2', 'Anxa2', 'Isyna1',
        'Fbln1', 'Islr', 'Slc13a3',
    ],
}
