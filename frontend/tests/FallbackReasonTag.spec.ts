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

describe('StockCard Fallback Reason tag', () => {
  it('renders fallback reason when legacy field is present', async () => {
    const wrapper = mount(StockCard, {
      props: {
        stock: {
          code: 'AAA',
          name: 'Foo Inc',
          marketType: 'US',
          analysisStatus: 'completed',
          analysis: 'test',
          // legacy backend field
          fallback_reason: '429' as any,
        } as any
      },
      global: { stubs }
    });

    expect(wrapper.text()).toContain('限速/配额');
  });

  it('renders fallback reason when modern field is present', async () => {
    const wrapper = mount(StockCard, {
      props: {
        stock: {
          code: 'BBB',
          name: 'Bar Inc',
          marketType: 'US',
          analysisStatus: 'completed',
          analysis: 'test',
          // modern typed field
          fallbackReason: 'empty_stream'
        } as any
      },
      global: { stubs }
    });

    expect(wrapper.text()).toContain('无流事件');
  });
});
