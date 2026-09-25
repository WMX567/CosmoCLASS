# CLASS-PT with self-interacting neutrinos

## Description 

This is a modified version of the [CLASS-PT code](https://github.com/Michalychforever/CLASS-PT) that computes the cosmology for self-interacting neutrino models as presented in [arXiv:2309.03941](https://arxiv.org/abs/2309.03941). The novel interaction among neutrinos and its impact on cosmological phenomena is fully controlled by a single parameter: `log10_G_eff_nu`. Please note that this code should be solely used in the synchronous gauge. For further details, see the following section. 

## Detailed implementation

We modified the Boltzmann equations in the perturbation module, `source/perturbations.c`, to account for self-interacting neutrinos as presented in [arXiv:2309.03941](https://arxiv.org/abs/2309.03941) and reference therein. Following [arXiv:1902.00534](https://arxiv.org/abs/1902.00534), we use precompute collision terms on a 5-point grid on the momentum space consistent with the grid used by the synchronous gauge perturbation solver. Such precompute terms can be found in the folder `neutrinos_collision_terms`. Although we also include an 11-point grid version that can be used to solve the cosmological perturbations in the Newtonian gauge, we highly discourage adopting such a gauge since the results are highly inaccurate. 

To avoid the stiffness of the Boltzmann equations when the mean free path of self-interacting neutrinos is much smaller than the Hubble horizon, we use tight-coupling approximation (TCA). We implemented such an approximation by exploiting the switch structure of CLASS. In particular, we added two new switches to control the TCA for self-interacting neutrinos: `nu_tca_on` and `nu_tca_off`. The boundaries of the TCA regime are delimited by the values of `start_small_k_at_tau_nu_over_tau_h`, `tight_coupling_trigger_tau_nu_over_tau_h`, and `tight_coupling_trigger_tau_nu_over_tau_k`. To avoid an overlap between the TCA and ultra-relativistic fluid approximation (UFA) for neutrinos, we delay the start of the UFA. Such delay is controlled by `full_hierarchy_trigger_tau_nu_over_tau_k`.

## Using the code

Please cite [arXiv:2309.03941](https://arxiv.org/abs/2309.03941) if you make use of this code. 

## Default values

```
  pba->log10_G_eff_nu=-12.;
  pba->G_eff_nu=0.;
  pba->interacting_nu=0.;

  ppr->start_small_k_at_tau_nu_over_tau_h = 2.e-5;
  ppr->tight_coupling_trigger_tau_nu_over_tau_h=0.001;
  ppr->tight_coupling_trigger_tau_nu_over_tau_k=0.001;
  ppr->full_hierarchy_trigger_tau_nu_over_tau_k=1.e4;

  ```
