l = "female,male,when_buying,buying,leasing,undecided,no_thought,my_choice,shared_choice,their_choice,me_driving,spouse_driving,kid_driving,other_driving,last_90,last_5mos,last_9mos,last_year,just_now,recognizing,researching,comparing,almost_chosen,chosen,r_family,r_newsize,r_safety,r_newer,r_deal,r_leaseup,r_needcar,r_tech,r_fuel,r_reliable,r_accident,r_luxury,r_bettercar,r_status,subcompact,smallcar,midsize,minivan,small_suv,midsize_suv,large_suv,small_pickup,large_pickup,sports,luxury,lux_suv,ultra_lux,perf_lux,cnt_considered_calc,electric,autonomous,cnt_considered_self,last_bought90d,last_bought1y,last_bought3y,last_bought5y,last_bought5y,last_boughtnever,never_mar,married,widowed,divorced,poor,middle,rich,1model,model"
l = l.split(",")
print(l)
st = ""
for i in l:
	st += "SUM(" + i + "), "

print(st)