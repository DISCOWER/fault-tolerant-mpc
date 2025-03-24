subs1 = {
    "j__00" : "j00",
    "j__11" : "j11",
    "j__22" : "j22",
    "k__1" : "k1",
    "k__2" : "k2",
    "k__3" : "k3",
    "omega__d" : "omd",
    "r" : "rNorm",
    "sqrNormt" : "sqrt",
    "q__u1" : "qu1",
    "q__u2" : "qu2",
    "q__u3" : "qu3",
    "q__u4" : "qu4",
    "q__u5" : "qu5",
    "q__u6" : "qu6",
    "`e__&omega;1`" : "eo1",
    "`e__&omega;2`" : "eo2",
    "`e__&omega;3`" : "eo3",
    "^" : "**"
}

expr1 = """(4*r^2*j__11*omega__d/j__00 - 4*r^2*j__22*omega__d/j__00 + 2*r^2*j__11^2*omega__d/j__00^2 + 2*r^2*j__22^2*omega__d/j__00^2 - 4*j__11*j__22*omega__d/j__00^2 + 2*r^2*omega__d + 2*j__11^2*omega__d/j__00^2 + 2*j__22^2*omega__d/j__00^2 - 4*r^2*j__11*j__22*omega__d/j__00^2)*`e__&omega;3`*`e__&omega;2`^2 
+ (r^2 - 2*r^2*j__11*j__22/j__00^2 + j__11^2/j__00^2 + j__22^2/j__00^2 - 2*r^2*j__22/j__00 + r^2*j__11^2/j__00^2 + r^2*j__22^2/j__00^2 - 2*j__11*j__22/j__00^2 + 2*r^2*j__11/j__00)*`e__&omega;3`^2*`e__&omega;2`^2 
+ (2*j__00^2*omega__d/j__11^2 + 2*j__22^2*omega__d/j__11^2 + 4*r^2*omega__d - 4*j__00*j__22*omega__d/j__11^2)*`e__&omega;3`*`e__&omega;1`^2 
+ (-2*j__00*j__22/j__11^2 + 2*r^2 + j__00^2/j__11^2 + j__22^2/j__11^2)*`e__&omega;3`^2*`e__&omega;1`^2 
+ (-2*j__00*j__11/j__22^2 + r^2*j__00^2/j__22^2 - 2*r^2*j__00/j__22 + r^2*j__11^2/j__22^2 + 2*r^2*j__11/j__22 + j__00^2/j__22^2 + j__11^2/j__22^2 + r^2 - 2*r^2*j__00*j__11/j__22^2)*`e__&omega;2`^2*`e__&omega;1`^2 
+ (-2*r^2*j__22*omega__d^2/j__00 + r^2*j__11^2*omega__d^2/j__00^2 + r^2*j__22^2*omega__d^2/j__00^2 - 2*j__11*j__22*omega__d^2/j__00^2 + 2*r^2*j__11*omega__d^2/j__00 + j__11^2*omega__d^2/j__00^2 + j__22^2*omega__d^2/j__00^2 + r^2*omega__d^2 - 2*r^2*j__11*j__22*omega__d^2/j__00^2)*`e__&omega;2`^2 
+ r^2*`e__&omega;1`^4 
+ r^2*`e__&omega;3`^4 
+ (-2*j__00*j__22*omega__d^2/j__11^2 + j__00^2*omega__d^2/j__11^2 + j__22^2*omega__d^2/j__11^2)*`e__&omega;1`^2 
+ 4*r^2*`e__&omega;3`^2*omega__d^2 
+ 4*r^2*`e__&omega;3`^3*omega__d"""
code1 = expr1

for k in subs1.keys():
    code1 = code1.replace(k, subs1[k])

# ============ End step 1 =====================================================================

subs2 = {
    "eo3*eo2**2 "    : "1 / ( 1 - (  (1 - k3)**1 * (1 - k2)**2             ))",
    "eo3**2*eo2**2 " : "1 / ( 1 - (  (1 - k3)**2 * (1 - k2)**2              ))",
    "eo3*eo1**2 "    : "1 / ( 1 - (  (1 - k3)**1 * (1 - k1)**2              ))",
    "eo3**2*eo1**2 " : "1 / ( 1 - (  (1 - k3)**2 * (1 - k1)**2              ))",
    "eo2**2*eo1**2 " : "1 / ( 1 - (  (1 - k2)**2 * (1 - k1)**2              ))",
    "eo2**2 "        : "1 / ( 1 - (  (1 - k2)**2              ))",
    "eo1**4 "        : "1 / ( 1 - (  (1 - k1)**4              ))",
    "eo3**4 "        : "1 / ( 1 - (  (1 - k3)**4              ))",
    "eo1**2 "        : "1 / ( 1 - (  (1 - k1)**2              ))",
    "eo3**2 "        : "1 / ( 1 - (  (1 - k3)**2              ))",
    "eo3**3"         : "1 / ( 1 - (  (1 - k3)**3              ))",
}

code = {
    "(4*rNorm**2*j11*omd/j00 - 4*rNorm**2*j22*omd/j00 + 2*rNorm**2*j11**2*omd/j00**2 + 2*rNorm**2*j22**2*omd/j00**2 - 4*j11*j22*omd/j00**2 + 2*rNorm**2*omd + 2*j11**2*omd/j00**2 + 2*j22**2*omd/j00**2 - 4*rNorm**2*j11*j22*omd/j00**2)" : "eo3*eo2**2 ",
    "(rNorm**2 - 2*rNorm**2*j11*j22/j00**2 + j11**2/j00**2 + j22**2/j00**2 - 2*rNorm**2*j22/j00 + rNorm**2*j11**2/j00**2 + rNorm**2*j22**2/j00**2 - 2*j11*j22/j00**2 + 2*rNorm**2*j11/j00)" : "eo3**2*eo2**2 ",
    "(2*j00**2*omd/j11**2 + 2*j22**2*omd/j11**2 + 4*rNorm**2*omd - 4*j00*j22*omd/j11**2)" : "eo3*eo1**2 ",
    "(-2*j00*j22/j11**2 + 2*rNorm**2 + j00**2/j11**2 + j22**2/j11**2)" : "eo3**2*eo1**2 ",
    "(-2*j00*j11/j22**2 + rNorm**2*j00**2/j22**2 - 2*rNorm**2*j00/j22 + rNorm**2*j11**2/j22**2 + 2*rNorm**2*j11/j22 + j00**2/j22**2 + j11**2/j22**2 + rNorm**2 - 2*rNorm**2*j00*j11/j22**2)" : "eo2**2*eo1**2 ",
    "(-2*rNorm**2*j22*omd**2/j00 + rNorm**2*j11**2*omd**2/j00**2 + rNorm**2*j22**2*omd**2/j00**2 - 2*j11*j22*omd**2/j00**2 + 2*rNorm**2*j11*omd**2/j00 + j11**2*omd**2/j00**2 + j22**2*omd**2/j00**2 + rNorm**2*omd**2 - 2*rNorm**2*j11*j22*omd**2/j00**2)" : "eo2**2 ",
    "rNorm**2" : "eo1**4 ",
    "rNorm**2" : "eo3**4 ",
    "(-2*j00*j22*omd**2/j11**2 + j00**2*omd**2/j11**2 + j22**2*omd**2/j11**2)" : "eo1**2 ",
    "4*rNorm**2*omd**2" : "eo3**2 ",
    "4*rNorm**2*omd" : "eo3**3",
}

full = ""

for k in code.keys():
    const_term = k
    error_zero = code[k]
    k_inf_sum = subs2[error_zero].replace(" ", "")
    full += " + " + const_term + "*" + error_zero + "*" + k_inf_sum

print(full)
print("WARNING: LOOK AT THE CODE IF YOU CHANGED SOMETHING, MAYBE THE RESULT WASNT UPDATED")
