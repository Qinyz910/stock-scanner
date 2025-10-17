module.exports = {
  root: true,
  env: {
    browser: true,
    es2023: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:vue/vue3-recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  parser: 'vue-eslint-parser',
  parserOptions: {
    parser: '@typescript-eslint/parser',
    extraFileExtensions: ['.vue'],
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  ignorePatterns: ['dist', 'coverage', 'node_modules'],
  rules: {
    'vue/multi-word-component-names': 'off',
  },
  overrides: [
    {
      files: ['**/__tests__/**/*.{ts,tsx}', '**/*.test.{ts,tsx}'],
      env: {
        vitest: true,
      },
    },
  ],
}
