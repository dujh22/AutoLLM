import React from 'react';
import { Timeline, App } from 'antd';
import { SaveOutlined } from '@ant-design/icons';
import { ISimpleData } from '@/pages/home/types.ts';
import dayjs from 'dayjs';
import styles from './styles.module.scss';

interface IDataProps {
  dataList: ISimpleData[];
}

const SimpleData: React.FC<IDataProps> = ({ dataList }) => {
  const { message } = App.useApp();

  const onSave = () => {
    message.info('复制到知识库，To be done...');
  };

  const items = dataList.map(item => {
    return {
      label:
      <div className={styles.timelineLabel}>
        <span>{dayjs(item.dateTime ? new Date(item.dateTime) : new Date()).format('YYYY-MM-DD')}</span>
         <SaveOutlined  onClick={() => onSave()} />
      </div>,

      children: (
        <div className={styles.item}>
          <div style={{ whiteSpace: 'pre-line' }}  dangerouslySetInnerHTML={{ __html: item.content }}></div>

        </div>
      ),
    };
  });

  return <Timeline mode="left" items={items} className={styles.timeline} />;
};

export default SimpleData;
