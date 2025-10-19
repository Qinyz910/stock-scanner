import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import StockCard from '@/components/StockCard.vue';

const stubs = {
  'n-card': {
    template: '<div class="n-card"><slot /></div>',
    props: ['bordered', 'class']
  },
  'n-divider': {
    template: '<div class="n-divider"><slot /></div>',
    props: ['dashed']
  },
  'n-tag': {
    template: '<span class="n-tag"><slot /></span>',
    props: ['type', 'size', 'round']
  },
  'n-icon': {
    template: '<i class="n-icon"><slot /></i>',
  },
  'n-button': {
    template: '<button class="n-button"><slot /></button>',
    props: ['size', 'type', 'secondary', 'round']
  },
  'n-space': {
    template: '<div class="n-space"><slot /></div>',
    props: ['align', 'wrap']
  }
};

describe('StockCard Risk and Confidence tags', () => {
  it('renders risk tag and low-confidence badge', async () => {
    const wrapper = mount(StockCard, {
      props: {
        stock: {
          code: 'AAA',
          name: 'Foo Inc',
          marketType: 'US',
          analysisStatus: 'completed',
          score: 80,
          analysis: 'test',
          risk_tag: 'high',
          confidence: 0.2,
          analysis_date: '2024-01-01'
        }
      },
      global: { stubs }
    });

    expect(wrapper.text()).toContain('高风险');
    expect(wrapper.text()).toContain('低置信度');
  });
});
