public class While01 {

	public static void main(String[] args) {
		int a=0;
		
		while(a<10) {
			a++;
			// variable의 위치에 따라 return value 확인
			// a = 9 -> 실행문에서 print 되는 값은 '10'
			System.out.print(a + "  ");
		}
		
		System.out.println();
		
		
		a = 0;
		while(true) {
			a++;
			
			if(a>10) break;
			// 제어문 위치에 따른 return value 확인
			
			System.out.print(a + "  ");
		}
		
	}
}
