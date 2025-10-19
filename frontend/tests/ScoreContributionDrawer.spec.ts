import { describe, it, expect, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import ScoreContributionDrawer from '@/components/ScoreContributionDrawer.vue';

const stubs = {
  'n-drawer': {
    template: '<div class="n-drawer"><slot /></div>',
    props: ['show', 'width', 'placement', 'trapFocus']
  },
  'n-drawer-content': {
    template: '<div class="n-drawer-content"><slot /><slot name="header" /></div>',
    props: ['title']
  },
  'n-empty': {
    template: '<div class="n-empty">empty: <slot /> <span>{{ description }}</span></div>',
    props: ['description', 'size']
  },
  'n-skeleton': {
    template: '<div class="n-skeleton">skeleton</div>',
    props: ['height', 'repeat', 'animated']
  },
  'n-tag': {
    template: '<span class="n-tag"><slot /></span>',
    props: ['type', 'size', 'round']
  },
  'n-divider': {
    template: '<div class="n-divider"><slot /></div>',
    props: ['dashed']
  },
  'n-button': {
    template: '<button @click="$emit(\'click\')"><slot /></button>',
    props: ['tertiary', 'size', 'loading']
  }
};

describe('ScoreContributionDrawer', () => {
  it('loads and renders contributions with confidence when opened', async () => {
    const loader = vi.fn(async () => ({
      items: [
        {
          code: 'AAA',
          score: 85,
          confidence: 0.9,
          contributions: {
            '因子A': 0.5,
            '因子B': -0.3,
            '因子C': 0.2
          }
        }
      ]
    }));

    const wrapper = mount(ScoreContributionDrawer, {
      props: {
        show: true,
        code: 'AAA',
        scoresLoader: loader
      },
      global: {
        stubs
      }
    });

    // 等待异步加载
    await new Promise((r) => setTimeout(r, 0));

    expect(loader).toHaveBeenCalled();
    expect(wrapper.text()).toContain('综合评分');
    expect(wrapper.text()).toContain('置信度');
    expect(wrapper.text()).toContain('因子A');
    expect(wrapper.text()).toContain('因子B');
  });
});
