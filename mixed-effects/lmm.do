/*
Notes: 
R2 in LMMs is calculated as decribed in Nakagawa & Schielzeth (2012) - https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.2041-210x.2012.00261.x

Why logging:
Since distributions of both the dependent and the independent variabls where skewed, we log-transformed all variables.
A linear model on the logarithmic scale then is equivalent to a multiplicative model on the untransformed scale (Gelman & Hill, 2007, p. 59)

Introduction to linear effects modeling:
https://stats.oarc.ucla.edu/other/mult-pkg/introduction-to-linear-mixed-models/

*/


*Get the data and pre-process
import delimited felix/aligned_words.csv, encoding(UTF-8) clear 

foreach var in et_importance length frequency lm_importance {
	gen double log_`var' = log(`var')
}

keep importance model corpus log* et_token

save aligned_words, replace


import delimited felix/human_words.csv, encoding(UTF-8) clear 
foreach var in et_importance length frequency  {
	gen double log_`var' = log(`var')
}


keep corpus log* et_token
save human_words, replace


****************************************************************
****************************************************************
****************************************************************
* Reproduce Table 4

clear 
set obs 1
gen corpus = ""
gen N = . /* Number of datapoints */
gen LMM = "" /* LMM without reffect ~ standard linear model vs. LMM with 'et_token' as reffect */
gen model = "" /* freq vs length vs freq+length */
gen double R2m = . /* coefficient of determination */
save Table4, replace

use human_words, clear

levelsof corpus
local corpora = r(levels)


quietly {
	foreach corpus of local corpora {
		foreach model in freq length both {
			foreach LMM in LMMwithout_reffect LMMwith_reffect {
				noisily di "`corpus' | `LMM' | `model'"
				
				if "`LMM'" == "LMMwithout_reffect" {
					local reffect
				}
				
				if "`LMM'" == "LMMwith_reffect" {
					local reffect `"et_token:"'
				}
				
				*-----------------------freq-----------------------*
				if "`model'" == "freq" {
						use if corpus == "`corpus'" using human_words, clear
						
						*Fit the model
						mixed log_et_importance log_frequency || `reffect' , reml
						predict double hat  
						sum hat
						
						*Collect the variance components and calculate R2m  <-- coefficient of determination
						local vc_fixed=r(Var)
						local N=e(N)
						matrix b = e(b)
						if "`LMM'" == "LMMwithout_reffect" {
						local vc_res = exp(b[1, 3])^2
						local R2m=`vc_fixed'/(`vc_fixed'+`vc_res')
						}
						if "`LMM'" == "LMMwith_reffect" {
						local vc_1   = exp(b[1, 3])^2
						local vc_res = exp(b[1, 4])^2
						local R2m=`vc_fixed'/(`vc_fixed'+`vc_1'+`vc_res')
						}	
					}
					*-----------------------length-----------------------*
				if "`model'" == "length" {
						use if corpus == "`corpus'" using human_words, clear
						
						*Fit the model
						mixed log_et_importance log_length || `reffect' , reml
						predict double hat  
						sum hat
						
						*Collect the variance components and calculate R2m  <-- coefficient of determination
						local vc_fixed=r(Var)
						local N=e(N)
						matrix b = e(b)
						if "`LMM'" == "LMMwithout_reffect" {
						local vc_res = exp(b[1, 3])^2
						local R2m=`vc_fixed'/(`vc_fixed'+`vc_res')
						}
						if "`LMM'" == "LMMwith_reffect" {
						local vc_1   = exp(b[1, 3])^2
						local vc_res = exp(b[1, 4])^2
						local R2m=`vc_fixed'/(`vc_fixed'+`vc_1'+`vc_res')
						}	
					}
					*-----------------------both-----------------------*
				if "`model'" == "both" {
						use if corpus == "`corpus'" using human_words, clear
						
						*Fit the model
						mixed log_et_importance log_frequ log_length || `reffect' , reml
						predict double hat  
						sum hat
						
						*Collect the variance components and calculate R2m  <-- coefficient of determination
						local vc_fixed=r(Var)
						local N=e(N)
						matrix b = e(b)
						if "`LMM'" == "LMMwithout_reffect" {
						local vc_res = exp(b[1, 4])^2
						local R2m=`vc_fixed'/(`vc_fixed'+`vc_res')
						}
						if "`LMM'" == "LMMwith_reffect" {
						local vc_1   = exp(b[1, 4])^2
						local vc_res = exp(b[1, 5])^2
						local R2m=`vc_fixed'/(`vc_fixed'+`vc_1'+`vc_res')
						}	
					}
			*Store results
			use Table4, clear
			local new = _N +1
			set obs `new'
			replace corpus = "`corpus'" in `new'
			replace N = `N' in `new'
			replace LMM = "`LMM'" in `new'
			replace model = "`model'" in `new'
			replace R2m = `R2m' in `new'
			drop if N == .
			save Table4, replace			
			}
		}
		
	}
}	

*Export 
export delimited using "Table4", delimiter(tab) replace
		
****************************************************************
****************************************************************
****************************************************************


