import importlib
import unittest

import numpy as np
from numba import njit

import nu_evolution as neutrino


CONFIGURATIONS = (
    (3.044, 2 * 3.044 / 3, 0.06, 2.7255),
    (3.046, 2 * 3.046 / 3, 0.06, 2.7255),
    (3.044, 2.0308, 0.06, 2.7255),
    (3.2, 2.0, 0.12, 2.8),
)
MODULE_NAMES = (
    "act", "act_planck", "early_lcdm", "planck", "planck_lens",
    "spt_planck_act",
)


def reference_evolution(config, z):
    m0, Omnu_h2 = neutrino.get_m0_and_Omnu_h2(*config)
    qs_sq = neutrino.compute_qs(m0)**2
    rho0 = neutrino.compute_rho0(m0)
    mz_sq = (m0 / (1.0 + z))**2
    f = np.sqrt(qs_sq[0] + mz_sq)
    density = neutrino.weights[0] * f
    numerator = neutrino.weights[0] / f
    for i in range(1, len(qs_sq)):
        f = np.sqrt(qs_sq[i] + mz_sq)
        density += neutrino.weights[i] * f
        numerator += neutrino.weights[i] / f
    return (
        Omnu_h2,
        (1.0 + z)**4 * density / rho0,
        (1 / 3) - (1 / 3) * mz_sq * numerator / density,
    )


class NeutrinoEvolutionTests(unittest.TestCase):
    def test_matches_previous_formulas_for_scalars_and_arrays(self):
        redshifts = np.concatenate(([0.0], np.logspace(-3, 7, 80)))
        for config in CONFIGURATIONS:
            Omnu_h2, Omnu_z, w_nu_z = neutrino.create_neutrino_evolution(*config)
            for z in (0.0, 1.0, 1100.0, 1e7, redshifts, redshifts.reshape(9, 9)):
                with self.subTest(config=config, shape=np.shape(z)):
                    expected_h2, expected_density, expected_w = reference_evolution(config, z)
                    self.assertEqual(Omnu_h2, expected_h2)
                    np.testing.assert_allclose(Omnu_z(z), expected_density, rtol=1e-14)
                    np.testing.assert_allclose(w_nu_z(z), expected_w, rtol=1e-14, atol=1e-16)
                    self.assertEqual(np.shape(Omnu_z(z)), np.shape(z))
                    self.assertEqual(np.shape(w_nu_z(z)), np.shape(z))
            self.assertAlmostEqual(Omnu_z(0.0), 1.0)
            self.assertAlmostEqual(w_nu_z(1e7), 1 / 3, places=9)

    def test_independent_configurations_in_nopython_callers(self):
        _, density_first, w_first = neutrino.create_neutrino_evolution(*CONFIGURATIONS[0])
        _, density_second, w_second = neutrino.create_neutrino_evolution(*CONFIGURATIONS[3])

        @njit
        def evaluate(z):
            return density_first(z), w_first(z), density_second(z), w_second(z)

        for z in (1100.0, np.array([0.0, 1.0, 1100.0])):
            actual = evaluate(z)
            expected_first = reference_evolution(CONFIGURATIONS[0], z)[1:]
            expected_second = reference_evolution(CONFIGURATIONS[3], z)[1:]
            for result, expected in zip(actual, expected_first + expected_second):
                np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-16)
        self.assertEqual(len(evaluate.nopython_signatures), 2)
        self.assertNotEqual(evaluate(1100.0)[0], evaluate(1100.0)[2])

    def test_zero_mass_limit(self):
        Omnu_h2, Omnu_z, w_nu_z = neutrino.create_neutrino_evolution(
            3.044, 2 * 3.044 / 3, 0.0, 2.7255
        )
        z = np.array([0.0, 1.0, 1100.0, 1e7])
        self.assertEqual(Omnu_h2, 0.0)
        np.testing.assert_allclose(Omnu_z(z), (1.0 + z)**4, rtol=1e-14)
        np.testing.assert_array_equal(w_nu_z(z), np.full_like(z, 1 / 3))

    def test_invalid_configuration(self):
        invalid = (
            (3.044, -1.0, 0.06, 2.7255),
            (3.044, 3.044, 0.06, 2.7255),
            (3.044, 4.0, 0.06, 2.7255),
            (3.044, 2.0, -0.06, 2.7255),
            (3.044, 2.0, 0.06, 0.0),
            (3.044, 2.0, 0.06, -2.7255),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                neutrino.create_neutrino_evolution(*config)
        for index in range(4):
            for value in (np.nan, np.inf, -np.inf):
                config = list(CONFIGURATIONS[0])
                config[index] = value
                with self.subTest(config=config), self.assertRaises(ValueError):
                    neutrino.create_neutrino_evolution(*config)

    def test_all_compression_modules_in_nopython_callers(self):
        for name in MODULE_NAMES:
            module = importlib.import_module(f"cmb.data_{name}_compression")
            density = module.Omnu_z
            equation_of_state = module.w_nu_z

            @njit
            def evaluate(z):
                return density(z), equation_of_state(z)

            config = (module.N_EFF, module.NU_REL, module.MNU_TOT, module.TCMB)
            for z in (0.0, np.array([0.0, 1.0, 1100.0, 1e7])):
                with self.subTest(module=name, shape=np.shape(z)):
                    expected_h2, expected_density, expected_w = reference_evolution(config, z)
                    actual_density, actual_w = evaluate(z)
                    self.assertEqual(module.Omnu_h2, expected_h2)
                    np.testing.assert_allclose(actual_density, expected_density, rtol=1e-14)
                    np.testing.assert_allclose(actual_w, expected_w, rtol=1e-14, atol=1e-16)
            self.assertEqual(len(evaluate.nopython_signatures), 2)

    def test_cmb_likelihood_regression(self):
        import cmb.cmb as model

        params = np.array([0.674, 0.0224, 0.12])
        model.cmb.set_HZ(model.Hz)
        likelihood, blobs = model.log_likelihood(params)
        self.assertAlmostEqual(likelihood, 13.246418658217703, places=9)
        self.assertEqual(blobs.shape, (5,))
        np.testing.assert_allclose(model.Hz(0.0, params), 67.4, rtol=1e-14)

        # Baselines use the same ACT+Planck H(z) for all six distance calculations.
        expected_chi2 = (
            5.646402028889981, 2.8942568935691915, 4.783859592830149,
            0.2428168838655553, 0.2322098624353651, 2.246383782989886,
        )
        expected_likelihood = (
            10.69521067114324, 13.246418658217703, 18.911554242766652,
            14.139178988457052, 14.271247005040868, 20.79661155145246,
        )
        for name, chi2, loglike in zip(MODULE_NAMES, expected_chi2, expected_likelihood):
            module = importlib.import_module(f"cmb.data_{name}_compression")
            with self.subTest(module=name):
                distances = module.cmb_distances(params[1], params[2], params)
                self.assertEqual(distances.shape, (3,))
                self.assertTrue(np.all(np.isfinite(distances)))
                self.assertAlmostEqual(module.chi2(params[1], params[2], params), chi2, places=9)
                self.assertAlmostEqual(
                    module.log_likelihood(params[1], params[2], params), loglike, places=9
                )


if __name__ == "__main__":
    unittest.main()
