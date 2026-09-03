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

  // Who sees a class marked `preview: true` in its workbook frontmatter. Nicknames,
  // because that is what the app knows about whoever is signed in. This is a cosmetic
  // gate: questions.json is public and holds every answer, so it keeps an unfinished
  // set off the students' screens, not out of their reach. Empty it, or delete the
  // frontmatter line and rebuild, to release a set to everyone.
  preview: ['Claude Shannon', 'Alan Turing'],
};
