// Both keys here are publishable keys. They are meant to be in a public repo —
// row-level security, not secrecy, is what protects the data.
window.COMP5423 = {
  envs: {
    dev: {
      url: 'https://fhbduoxqktowbpruqqhh.supabase.co',
      key: 'sb_publishable_laRSaPegiz1DvhBmpqnqbw_zMzgY75K',
    },
    // prod: filled in on Aug 29
  },
  defaultEnv: 'dev',        // flip to 'prod' on Aug 30
};
