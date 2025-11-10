using Pkg
Pkg.add("ThreeBodyTB")
using ThreeBodyTB

println("run test")

function go() 
    outfile = open("dft_3d_cubic_lattice_param_a.json", "w")
    write(outfile, "{\n")
    write(outfile, "  \"train\":{},\n")
    write(outfile, "  \"test\":{\n")

    ang = 0.529177

    #loop over test set
    for poscar in [        "JVASP-91",
                           "JVASP-148350",
                           "JVASP-14658",
                           "JVASP-120123",
                           "JVASP-1327",
                           "JVASP-1408",
                           "JVASP-1393",
                           "JVASP-1174",
                           "JVASP-1177",
                           "JVASP-1183",
                           "JVASP-1186",
                           "JVASP-1189",
                           "JVASP-116",
                           "JVASP-8003",
                           "JVASP-1192",
                           "JVASP-23",
                           "JVASP-7923",
                           "JVASP-1702",
                           "JVASP-150249",
                           "JVASP-150058",
                           "JVASP-7836",
                           "JVASP-1312",
                           "JVASP-120284",
                           "JVASP-23864",
                           "JVASP-113667",
                           "JVASP-8563",
                           "JVASP-118783",
                           "JVASP-23862",
                           "JVASP-1951",
                           "JVASP-1996",
                           "JVASP-20309",
                           "JVASP-1145",
                           "JVASP-1993",
                           "JVASP-1942",
                           "JVASP-21717",
                           "JVASP-113493",
                           "JVASP-1921",
                           "JVASP-1945",
                           "JVASP-14615",
                           "JVASP-104984",
                           "JVASP-14630",
                           "JVASP-14648",
                           "JVASP-14610",
                           "JVASP-78335",
                           "JVASP-14606",
                           "JVASP-79561",
                           "JVASP-14607",
                           "JVASP-825",
                           "JVASP-961",
                           "JVASP-78326",
                           "JVASP-14818",
                           "JVASP-14750",
                           "JVASP-14741",
                           "JVASP-19679",
                           "JVASP-15075",
                           "JVASP-19896",
                           "JVASP-35790",
                           "JVASP-19767",
                           "JVASP-18916",
                           "JVASP-15086",
                           "JVASP-1127",
                           "JVASP-8082",
                           "JVASP-7848",
                           "JVASP-117995"]
        try
            if isfile("POSCAR."*poscar)
                #println("$poscar ", isfile("POSCAR_"*poscar))
                
                c = makecrys("POSCAR.$poscar")
                c_std = ThreeBodyTB.Symmetry.get_standard_crys(c)
                cfinal, ret = relax_structure(c_std)
                c_conv = ThreeBodyTB.Symmetry.get_standard_crys(cfinal, to_primitive=false)
                write(outfile, "      \"$(poscar)\": $(c_conv.A[1,1]*ang), \n")
            else
                println("missing $poscar --------------------------------------------------------------------------------")
            end
        catch
            println("fail $poscar --------------------------------------------------------------------------------")
        end
    end
    write(outfile, "  }\n")
    write(outfile, "}\n")
    close(outfile)
end

go()

